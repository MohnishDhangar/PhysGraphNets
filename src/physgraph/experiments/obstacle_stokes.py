"""Finite-domain obstacle Stokes studies; no renderer or learned fields."""
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from physgraph.physics.obstacle_stokes import ObstacleStokesCase
from physgraph.mesh.annulus import annulus_mesh
from physgraph.mesh.cylinder import cylinder_mesh
from physgraph.mesh.airfoil import airfoil_mesh, profile_area
from physgraph.mesh.quality import mesh_quality
from physgraph.fem.stokes import solve_stokes
from physgraph.evaluation.stokes import stokes_errors
from physgraph.evaluation.obstacle_stokes import obstacle_metrics
from physgraph.io.manifest import write_manifest


def obstacle_mesh(case, level):
    if type(level) is not int or level < 2:
        raise ValueError('obstacle resolution level must be an integer >= 2')
    if case.name == 'annular_cylinder':
        return annulus_mesh(level, 16*level, case.radius, case.outer_radius, grading=1.5)
    if case.name == 'channel_cylinder':
        return cylinder_mesh(level, 4*level, case.radius,
            bounds=(-case.half_length, case.half_length, -case.half_height, case.half_height), grading=1.5)
    return airfoil_mesh(level, 16*level, thickness=case.thickness, chord=case.chord,
                        farfield_radius=case.outer_radius, grading=1.5, angular_bias=case.airfoil_angular_bias)


def evaluate_obstacle(case, level, *, external=False):
    mesh = obstacle_mesh(case, level)
    quality = mesh_quality(mesh)
    if quality['holes'] != 1 or quality['min_scaled_quality'] < 1e-4:
        raise ValueError('obstacle mesh must have one hole and scaled quality >= 1e-4')
    result = solve_stokes(mesh, case.zero, case.boundary_velocity, viscosity=case.viscosity,
        viscous_form='symmetric_gradient', dirichlet_tags=case.dirichlet_tags, tractions=case.tractions)
    row = {'case': case.name, 'level': level, 'parameters': asdict(case), 'mesh_quality': quality,
           **result.diagnostics, **obstacle_metrics(result, case)}
    target_area = (profile_area(case.thickness, case.chord) if case.name == 'airfoil' else np.pi*case.radius**2)
    row['relative_obstacle_area_error'] = abs(-min(quality['boundary_signed_areas'])-target_area)/target_area
    if case.name == 'annular_cylinder':
        row['analytical_errors_on_polygon'] = stokes_errors(result, case)
        row['analytical_drag'] = case.analytical_drag
        row['relative_drag_error'] = abs(row['reaction_force_on_obstacle'][0]/case.analytical_drag-1)
    if external:
        from physgraph.verification.stokes import compare_stokes
        row['fenicsx'] = compare_stokes(result, case)
    return result, row


def run_obstacle_stokes(*, output, levels=(4, 8, 16), external=False,
                       cases=('annular_cylinder', 'channel_cylinder', 'airfoil'), domain_study=True,
                       airfoil_angular_bias=.2):
    levels = list(levels)
    if len(levels) < 3 or any(type(n) is not int or n < 2 for n in levels) or any(b != 2*a for a, b in zip(levels, levels[1:])):
        raise ValueError('require at least three successively doubled levels >= 2')
    if not cases or len(set(cases)) != len(cases):
        raise ValueError('require distinct nonempty case names')
    fixtures = [ObstacleStokesCase(name, airfoil_angular_bias=airfoil_angular_bias) for name in cases]
    dest = Path(output)
    if dest.exists() and (not dest.is_dir() or any(dest.iterdir())):
        raise ValueError('obstacle output must be empty')
    dest.mkdir(parents=True, exist_ok=True)
    gates = dict(residual=1e-9, no_slip=1e-12, relative_net_flux=1e-9,
                 relative_force_balance=1e-8, external=1e-9, positive_drag=0.,
                 finest_relative_drag_change=.03, finest_relative_force_disagreement=.05,
                 finest_relative_lift=.02, finest_analytical_drag_error=.02,
                 finest_annular_velocity_l2_rate=1.5, finest_annular_pressure_l2_rate=1.5)
    config = dict(levels=levels, cases=[asdict(c) for c in fixtures], external=external,
                  domain_study=domain_study, acceptance=gates, units='dimensionless',
                  method='CPU P2/P1 symmetric-gradient Stokes; straight polygon facets',
                  force='negative Dirichlet reaction on closed obstacle, checked against surface stress',
                  scope='finite domains; no inertial term; no infinite-domain drag claim')
    started = perf_counter()
    metrics = {'status': 'running', 'rows': [], 'domain_rows': []}
    write_manifest(dest/'manifest.json', config, metrics, 0.)

    def checks(row, case):
        scale = case.viscosity*case.speed
        answer = {'residual': row['relative_residual'] <= gates['residual'],
                  'no_slip': row['max_obstacle_speed'] <= gates['no_slip'],
                  'flux': abs(row['boundary_integrals']['net_flux'])/case.speed <= gates['relative_net_flux'],
                  'force_balance': row['algebraic_force_balance']/scale <= gates['relative_force_balance'],
                  'positive_drag': row['reaction_force_on_obstacle'][0] > gates['positive_drag']}
        if external:
            answer['fenicsx'] = row['fenicsx']['passed']
        return answer

    def save(result, row, label):
        mesh = result.space.mesh
        np.savez_compressed(dest/f'{label}.npz', nodes=mesh.nodes, triangles=mesh.triangles,
            boundary_edges=mesh.boundary_edges, boundary_edge_tags=mesh.boundary_edge_tags,
            velocity_nodes=result.space.points, velocity_cells=result.space.cells,
            velocity=result.velocity, pressure=result.pressure, velocity_reactions=result.velocity_reactions,
            metadata=np.array(json.dumps({'schema_version': 1, 'artifact': 'obstacle_stokes', **row['parameters']})))

    try:
        for case in fixtures:
            previous = None
            for level in levels:
                result, row = evaluate_obstacle(case, level, external=external)
                accepted = checks(row, case)
                if previous is not None:
                    row['relative_drag_change'] = abs(row['reaction_force_on_obstacle'][0]-previous['reaction_force_on_obstacle'][0])/abs(row['reaction_force_on_obstacle'][0])
                    if case.name == 'annular_cylinder':
                        for key in ('velocity_l2', 'pressure_l2'):
                            row[key+'_rate'] = float(np.log2(previous['analytical_errors_on_polygon'][key]/row['analytical_errors_on_polygon'][key]))
                    if level == levels[-1]:
                        accepted.update(drag_refinement=row['relative_drag_change'] <= gates['finest_relative_drag_change'],
                            force_estimators=row['relative_force_disagreement'] <= gates['finest_relative_force_disagreement'],
                            symmetry=row['relative_lift'] <= gates['finest_relative_lift'])
                        if case.name == 'annular_cylinder':
                            accepted['analytical_drag'] = row['relative_drag_error'] <= gates['finest_analytical_drag_error']
                            for key in ('velocity_l2', 'pressure_l2'):
                                accepted[key+'_rate'] = row[key+'_rate'] >= gates['finest_annular_'+key+'_rate']
                row.update(checks=accepted, passed=bool(all(accepted.values())))
                metrics['rows'].append(row)
                save(result, row, f'{case.name}_{level}')
                previous = row
                write_manifest(dest/'manifest.json', config, metrics, perf_counter()-started)
        # Distinct bounded problems: compare each radius to its own exact solution.
        # These are not estimates of a nonexistent uniform-flow infinite Stokes limit.
        if domain_study and 'annular_cylinder' in cases:
            for radius in (1.5, 3.):
                case = ObstacleStokesCase(outer_radius=radius)
                previous = None
                for level in levels[-2:]:
                    result, row = evaluate_obstacle(case, level, external=external)
                    accepted = checks(row, case)
                    if previous is not None:
                        row['relative_drag_change'] = abs(row['reaction_force_on_obstacle'][0]-previous['reaction_force_on_obstacle'][0])/abs(row['reaction_force_on_obstacle'][0])
                        accepted['analytical_drag'] = row['relative_drag_error'] <= gates['finest_analytical_drag_error']
                        accepted['drag_refinement'] = row['relative_drag_change'] <= gates['finest_relative_drag_change']
                    row.update(checks=accepted, passed=bool(all(accepted.values())))
                    metrics['domain_rows'].append(row)
                    save(result, row, f'annular_radius_{radius}_{level}')
                    previous = row
                    write_manifest(dest/'manifest.json', config, metrics, perf_counter()-started)
        metrics['status'] = 'passed' if all(row['passed'] for row in metrics['rows']+metrics['domain_rows']) else 'failed_acceptance'
    except Exception as exc:
        metrics.update(status='failed', reason=f'{type(exc).__name__}: {exc}')
        write_manifest(dest/'manifest.json', config, metrics, perf_counter()-started)
        raise
    return write_manifest(dest/'manifest.json', config, metrics, perf_counter()-started)
