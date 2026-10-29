"""Declared manufactured transient-flow qualification; no renderer or training."""
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from physgraph.mesh.structured import tagged_unit_square_mesh
from physgraph.physics.navier_stokes import NavierStokesCase
from physgraph.fem.navier_stokes import solve_navier_stokes
from physgraph.evaluation.navier_stokes import trajectory_metrics, closed_wall_energy
from physgraph.io.manifest import write_manifest


def manufactured_trajectory(case, n, dt, steps, scheme, traction_tag=None):
    mesh = tagged_unit_square_mesh(n)
    return solve_navier_stokes(mesh, case.forcing, case.velocity, lambda x: case.velocity(x, 0),
        viscosity=case.viscosity, dt=dt, steps=steps, scheme=scheme,
        dirichlet_tags=None if traction_tag is None else set(mesh.boundary_edge_tags)-{traction_tag},
        tractions=None if traction_tag is None else {traction_tag: case.traction})


def save_trajectory(path, trajectory, metadata):
    mesh = trajectory.space.mesh
    np.savez_compressed(path, nodes=mesh.nodes, triangles=mesh.triangles,
        boundary_edges=mesh.boundary_edges, boundary_edge_tags=mesh.boundary_edge_tags,
        velocity_nodes=trajectory.space.points, velocity_cells=trajectory.space.cells,
        times=trajectory.times, velocity=trajectory.velocity, pressure_times=trajectory.times[1:],
        pressure=trajectory.pressure, velocity_reactions=trajectory.reactions,
        metadata=np.array(json.dumps({'schema_version': 1, 'artifact': 'transient_navier_stokes',
            'viscosity': trajectory.viscosity, 'scheme': trajectory.scheme,
            'dirichlet_tags': trajectory.dirichlet_tags,
            'initial_pressure': 'absent; projection multiplier is not physical pressure', **metadata})))


def run_navier_stokes(*, output, sizes=(4, 8, 16), time_steps=(8, 16, 32), external=False):
    for name, values in (('sizes', sizes), ('time_steps', time_steps)):
        if (len(values) < 3 or any(type(n) is not int or n < 2 for n in values)
                or any(b != 2*a for a, b in zip(values, values[1:]))):
            raise ValueError(f'{name} requires at least three successively doubled integers >= 2')
    dest = Path(output)
    if dest.exists() and (not dest.is_dir() or any(dest.iterdir())):
        raise ValueError('transient output must be empty')
    dest.mkdir(parents=True, exist_ok=True)
    gates = dict(residual=5e-11, boundary=1e-12, projection=1e-10, weak_divergence=1e-10,
        net_flux=1e-10, force_balance=1e-9, pressure_gauge=1e-10, patch=1e-9,
        time_be_rate=.85, time_bdf2_rate=1.75, spatial_velocity_l2_rate=2.5,
        spatial_velocity_h1_seminorm_rate=1.8, spatial_pressure_l2_rate=1.8,
        spatial_divergence_l2_rate=1.8, energy_balance=1e-9, energy_increase=1e-12, external=1e-9)
    config = dict(sizes=list(sizes), time_steps=list(time_steps), external=external,
        acceptance=gates, method='CPU P2/P1 fully implicit advective Navier-Stokes; Newton with backtracking',
        stress='symmetric_gradient', temporal='BE and constant-step BDF2 with one BE startup step',
        domain='unit square; manufactured full velocity and exact traction boundaries',
        viscosity=.1, assembly_quadrature_order=9, error_quadrature_order=9,
        scope='manufactured verification and closed-wall decay; obstacle wakes not qualified')
    started = perf_counter()
    metrics = {'status': 'running', 'rows': [], 'energy_rows': []}
    write_manifest(dest/'manifest.json', config, metrics, 0.)

    def evaluate(phase, case, n, dt, steps, scheme, tag, previous=None, finest=False):
        result = manufactured_trajectory(case, n, dt, steps, scheme, tag)
        row = {'phase': phase, 'case': asdict(case), 'n': n, 'dt': dt, 'steps': steps,
               'scheme': scheme, 'traction_tag': tag, **trajectory_metrics(result, case)}
        m = row['maxima']
        checks = dict(residual=m['relative_residual'] <= gates['residual'],
            boundary=m['max_boundary_error'] <= gates['boundary'],
            weak_divergence=m['weak_divergence_residual'] <= gates['weak_divergence'],
            flux=max(m['boundary_flux'], m['net_integrated_flux']) <= gates['net_flux'],
            force_balance=m['algebraic_force_balance'] <= gates['force_balance'],
            initial_projection=row['initial']['projection_residual'] <= gates['projection'])
        if tag is None:
            checks['pressure_gauge'] = m['pressure_mean'] <= gates['pressure_gauge']
        if phase == 'patch':
            checks['patch'] = max(max(frame[key] for key in row['final_errors']) for frame in row['frames']) <= gates['patch']
            checks['pressure_level'] = m['pressure_mean'] <= gates['patch']
        if previous is not None:
            keys = ('velocity_l2', 'pressure_l2') if phase == 'temporal' else tuple(row['final_errors'])
            row['rates'] = {key: float(np.log2(previous['final_errors'][key]/max(row['final_errors'][key], 1e-30))) for key in keys}
            if finest:
                for key, rate in row['rates'].items():
                    target = gates['time_'+scheme+'_rate'] if phase == 'temporal' else gates['spatial_'+key+'_rate']
                    checks[key+'_rate'] = rate >= target
        if external:
            from physgraph.verification.navier_stokes import compare_navier_stokes
            row['fenicsx'] = compare_navier_stokes(result, case, gates['external'])
            checks['fenicsx'] = row['fenicsx']['passed']
        row.update(checks=checks, passed=bool(all(checks.values())))
        metrics['rows'].append(row)
        label = f'{phase}_{case.name}_{scheme}_{tag or "full"}_n{n}_steps{steps}'
        save_trajectory(dest/f'{label}.npz', result, {key: row[key] for key in
            ('phase', 'case', 'n', 'dt', 'steps', 'traction_tag')})
        write_manifest(dest/'manifest.json', config, metrics, perf_counter()-started)
        return row

    try:
        for scheme in ('be', 'bdf2'):
            for tag in (None, 'left', 'right'):
                case = NavierStokesCase('accelerating_shear', time_profile='linear', pressure_offset=0. if tag is None else 2.5)
                evaluate('patch', case, 4, .05, 4, scheme, tag)
            for tag in (None, 'right'):
                previous = None
                case = NavierStokesCase('accelerating_shear', pressure_offset=0. if tag is None else 2.5)
                for steps in time_steps:
                    previous = evaluate('temporal', case, 4, .4/steps, steps, scheme, tag,
                                        previous=previous, finest=steps == time_steps[-1])
        # Linear amplitude has no BE/BDF2 consistency defect: this isolates spatial
        # refinement without supplying an analytical history to numerical steps.
        for name in ('vortex', 'translated_vortex'):
            for tag in (None, 'right'):
                previous = None
                case = NavierStokesCase(name, time_profile='linear', pressure_offset=0. if tag is None else 2.5)
                for n in sizes:
                    previous = evaluate('spatial', case, n, .025, 4, 'bdf2', tag,
                                        previous=previous, finest=n == sizes[-1])
        for scheme in ('be', 'bdf2'):
            case = NavierStokesCase('vortex')
            zero = lambda x,t: np.zeros_like(x)
            result = solve_navier_stokes(tagged_unit_square_mesh(6), zero, zero,
                lambda x: case.velocity(x, 0), dt=.025, steps=20, scheme=scheme)
            row = {'scheme': scheme, 'n': 6, 'dt': .025, 'steps': 20,
                   'case': 'unforced_closed_wall_decay', **closed_wall_energy(result)}
            row['checks'] = dict(energy_balance=row['max_absolute_balance'] <= gates['energy_balance'],
                                 energy_decay=row['max_energy_increase'] <= gates['energy_increase'])
            row['passed'] = bool(all(row['checks'].values()))
            metrics['energy_rows'].append(row)
            save_trajectory(dest/f'energy_{scheme}.npz', result, {'case': row['case'], 'n': 6, 'dt': .025, 'steps': 20})
        metrics['status'] = 'passed' if all(r['passed'] for r in metrics['rows']+metrics['energy_rows']) else 'failed_acceptance'
    except Exception as exc:
        metrics.update(status='failed', reason=f'{type(exc).__name__}: {exc}')
        write_manifest(dest/'manifest.json', config, metrics, perf_counter()-started)
        raise
    return write_manifest(dest/'manifest.json', config, metrics, perf_counter()-started)
