"""Reproducible Stokes patch tests and mesh refinement; no viewer output."""
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from physgraph.mesh.structured import unit_square_mesh
from physgraph.physics.stokes import StokesCase
from physgraph.fem.stokes import solve_stokes
from physgraph.evaluation.stokes import stokes_errors
from physgraph.io.manifest import write_manifest


def run_stokes(*, output, sizes=(4, 8, 16), viscosity=1., external=False):
    sizes = list(sizes)
    if len(sizes) < 3 or any(type(n) is not int or n < 2 for n in sizes) or any(b != 2*a for a, b in zip(sizes, sizes[1:])):
        raise ValueError('require at least three successively doubled integer sizes >= 2')
    StokesCase(viscosity=viscosity)
    dest = Path(output)
    if dest.exists() and (not dest.is_dir() or any(dest.iterdir())):
        raise ValueError('Stokes output must be empty')
    dest.mkdir(parents=True, exist_ok=True)
    gates = {'patch_max_error': 1e-10, 'residual': 1e-9, 'boundary': 1e-12,
             'pressure_mean': 1e-10, 'weak_divergence': 1e-10,
             'finest_velocity_l2_rate': 2.5, 'finest_velocity_h1_rate': 1.8,
             'finest_pressure_l2_rate': 1.8, 'finest_divergence_rate': 1.8}
    config = dict(sizes=sizes, viscosity=viscosity, external=external,
        method='P2/P1 Taylor-Hood, CPU float64, mean-zero pressure',
        boundary='full prescribed velocity; no traction outlet', domain='unit square',
        units='dimensionless', acceptance=gates, error_quadrature_order=9, assembly_quadrature_order=6)
    started = perf_counter()
    metrics = {'status': 'running', 'rows': []}
    write_manifest(dest/'manifest.json', config, metrics, 0.)
    try:
        passed = True
        for name in ('affine', 'channel', 'vortex', 'translated_vortex'):
            previous = None
            for n in sizes:
                case = StokesCase(name, viscosity)
                result = solve_stokes(unit_square_mesh(n), case.forcing, case.velocity, viscosity=viscosity)
                row = {'case': name, 'n': n, **result.diagnostics, **stokes_errors(result, case)}
                checks = [row['relative_residual'] <= gates['residual'],
                          row['max_boundary_error'] <= gates['boundary'],
                          abs(row['pressure_mean']) <= gates['pressure_mean'],
                          row['weak_divergence_residual'] <= gates['weak_divergence']]
                if name in ('affine', 'channel'):
                    checks.append(max(row[k] for k in ('velocity_l2', 'velocity_h1_seminorm', 'pressure_l2', 'divergence_l2')) <= gates['patch_max_error'])
                elif previous is not None:
                    for key, gate in [('velocity_l2', 'finest_velocity_l2_rate'), ('velocity_h1_seminorm', 'finest_velocity_h1_rate'),
                                      ('pressure_l2', 'finest_pressure_l2_rate'), ('divergence_l2', 'finest_divergence_rate')]:
                        row[key+'_rate'] = float(np.log2(previous[key]/max(row[key], 1e-30)))
                        if n == sizes[-1]:
                            checks.append(row[key+'_rate'] >= gates[gate])
                if external:
                    from physgraph.verification.stokes import compare_stokes
                    row['fenicsx'] = compare_stokes(result, case)
                    checks.append(row['fenicsx']['passed'])
                row['passed'] = bool(all(checks))
                passed &= row['passed']
                metrics['rows'].append(row)
                previous = row
                np.savez_compressed(dest/f'{name}_{n}.npz', nodes=result.space.mesh.nodes,
                    triangles=result.space.mesh.triangles, velocity_nodes=result.space.points,
                    velocity_cells=result.space.cells, velocity=result.velocity, pressure=result.pressure,
                    metadata=np.array(json.dumps({'schema_version': 1, 'case': name, **config})))
        metrics['status'] = 'passed' if passed else 'failed_acceptance'
    except Exception as exc:
        metrics.update(status='failed', reason=f'{type(exc).__name__}: {exc}')
        write_manifest(dest/'manifest.json', config, metrics, perf_counter()-started)
        raise
    return write_manifest(dest/'manifest.json', config, metrics, perf_counter()-started)
