"""Manufactured velocity/traction Stokes qualification with physical load checks."""
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from physgraph.mesh.structured import tagged_unit_square_mesh
from physgraph.physics.stokes import StokesCase
from physgraph.fem.stokes import solve_stokes
from physgraph.evaluation.stokes import stokes_errors, stokes_boundary_integrals
from physgraph.io.manifest import write_manifest


def run_mixed_stokes(*, output, sizes=(4, 8, 16), viscosity=.3, pressure_offset=2.5, external=False):
    sizes = list(sizes)
    if len(sizes) < 3 or any(type(n) is not int or n < 2 for n in sizes) or any(b != 2*a for a, b in zip(sizes, sizes[1:])):
        raise ValueError('require at least three successively doubled integer sizes >= 2')
    StokesCase(viscosity=viscosity, pressure_offset=pressure_offset)
    dest = Path(output)
    if dest.exists() and (not dest.is_dir() or any(dest.iterdir())):
        raise ValueError('mixed Stokes output must be empty')
    dest.mkdir(parents=True, exist_ok=True)
    gates = dict(patch_error=1e-10, residual=1e-9, boundary=1e-12, net_flux=1e-10,
                 weak_divergence=1e-10, algebraic_force_balance=1e-10,
                 velocity_l2_rate=2.5, velocity_h1_seminorm_rate=1.8,
                 pressure_l2_rate=1.8, divergence_l2_rate=1.8,
                 traction_l2_error_rate=1.2, finest_tag_force_error=.01)
    config = dict(sizes=sizes, viscosity=viscosity, pressure_offset=pressure_offset,
        external=external, method='P2/P1 Taylor-Hood, symmetric-gradient stress',
        boundary='one tagged traction side (left/right), prescribed velocity elsewhere',
        pressure_constraint='traction; no mean subtraction or pin', domain='unit square',
        units='dimensionless; pressure and loads divided by density',
        error_quadrature_order=9, assembly_quadrature_order=6, acceptance=gates)
    started = perf_counter()
    metrics = {'status': 'running', 'rows': []}
    write_manifest(dest/'manifest.json', config, metrics, 0.)
    try:
        for name in ('affine', 'channel', 'vortex', 'translated_vortex'):
            for traction_tag in ('right', 'left'):
                previous = None
                for n in sizes:
                    case = StokesCase(name, viscosity, pressure_offset)
                    mesh = tagged_unit_square_mesh(n)
                    selected = set(mesh.boundary_edge_tags)-{traction_tag}
                    result = solve_stokes(mesh, case.forcing, case.velocity, viscosity=viscosity,
                        viscous_form='symmetric_gradient', dirichlet_tags=selected, tractions={traction_tag: case.traction})
                    boundary = stokes_boundary_integrals(result, case)
                    row = {'case': name, 'traction_tag': traction_tag, 'n': n,
                           **result.diagnostics, **stokes_errors(result, case), 'boundary_integrals': boundary,
                           'traction_l2_error': boundary['traction_l2_error']}
                    checks = {'residual': row['relative_residual'] <= gates['residual'],
                        'boundary': row['max_boundary_error'] <= gates['boundary'],
                        'net_flux': abs(boundary['net_flux']) <= gates['net_flux'],
                        'weak_divergence': row['weak_divergence_residual'] <= gates['weak_divergence'],
                        'force_balance': row['algebraic_force_balance'] <= gates['algebraic_force_balance']}
                    if name in ('affine', 'channel'):
                        checks['patch_errors'] = max(row[k] for k in ('velocity_l2', 'velocity_h1_seminorm',
                            'pressure_l2', 'divergence_l2', 'traction_l2_error')) <= gates['patch_error']
                        checks['pressure_level'] = abs(row['pressure_mean']-pressure_offset) <= gates['patch_error']
                        checks['patch_load_balance'] = boundary['stress_force_balance'] <= gates['patch_error']
                        if name == 'channel':
                            checks['channel_fluxes'] = (abs(boundary['by_tag']['left']['flux']+1/6) <= gates['net_flux']
                                and abs(boundary['by_tag']['right']['flux']-1/6) <= gates['net_flux'])
                    elif previous is not None:
                        for key in ('velocity_l2', 'velocity_h1_seminorm', 'pressure_l2', 'divergence_l2', 'traction_l2_error'):
                            rate = float(np.log2(previous[key]/max(row[key], 1e-30)))
                            row[key+'_rate'] = rate
                            if n == sizes[-1]:
                                checks[key+'_rate'] = rate >= gates[key+'_rate']
                        if n == sizes[-1]:
                            checks['tag_force_accuracy'] = boundary['max_tag_force_error'] <= gates['finest_tag_force_error']
                    if external:
                        from physgraph.verification.stokes import compare_stokes
                        row['fenicsx'] = compare_stokes(result, case)
                        checks['fenicsx'] = row['fenicsx']['passed']
                    row['checks'], row['passed'] = checks, bool(all(checks.values()))
                    metrics['rows'].append(row)
                    previous = row
                    np.savez_compressed(dest/f'{name}_{traction_tag}_{n}.npz', nodes=mesh.nodes,
                        triangles=mesh.triangles, boundary_edges=mesh.boundary_edges,
                        boundary_edge_tags=mesh.boundary_edge_tags, velocity_nodes=result.space.points,
                        velocity_cells=result.space.cells, velocity=result.velocity, pressure=result.pressure,
                        metadata=np.array(json.dumps({'artifact': 'stokes_mixed_reference', 'schema_version': 1,
                            'case': name, 'traction_tag': traction_tag, 'dirichlet_tags': sorted(selected), **config})))
                    write_manifest(dest/'manifest.json', config, metrics, perf_counter()-started)
        metrics['status'] = 'passed' if all(row['passed'] for row in metrics['rows']) else 'failed_acceptance'
    except Exception as exc:
        metrics.update(status='failed', reason=f'{type(exc).__name__}: {exc}')
        write_manifest(dest/'manifest.json', config, metrics, perf_counter()-started)
        raise
    return write_manifest(dest/'manifest.json', config, metrics, perf_counter()-started)
