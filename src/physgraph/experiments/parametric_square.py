"""Convergence and integration sensitivity for the richer graph-label family."""
from pathlib import Path
from time import perf_counter
import numpy as np

from physgraph.fem.elliptic import solve
from physgraph.fem.error import field_errors
from physgraph.io.manifest import write_manifest
from physgraph.mesh.structured import unit_square_mesh
from physgraph.physics.parametric_square import square_case, DEFAULTS, BOUNDARY_MODES


def verify_family(*, output, resolutions=(8, 16, 32)):
    if (len(resolutions) < 2 or any(type(n) is not int or n < 2 for n in resolutions)
            or any(a >= b for a, b in zip(resolutions, resolutions[1:]))):
        raise ValueError('require at least two strictly increasing resolutions >=2')
    dest = Path(output)
    if dest.exists() and (not dest.is_dir() or any(dest.iterdir())):
        raise ValueError('verification output must be empty')
    dest.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    config = {'family': 'parametric_square', 'physics_defaults': DEFAULTS,
              'resolutions': list(resolutions), 'boundary_modes': list(BOUNDARY_MODES),
              'assembly_orders': [7, 9], 'error_orders': [5, 7],
              'minimum_l2_rate': 1.8, 'minimum_h1_rate': .9,
              'max_quadrature_nodal_difference': 1e-8, 'max_error_integration_difference': 1e-7}
    rows, rates = [], []
    for mode in BOUNDARY_MODES:
        case = square_case(boundary_mode=mode)
        previous = None
        for n in resolutions:
            mesh = unit_square_mesh(n)
            baseline, higher = (solve(mesh, case.problem, order) for order in (7, 9))
            errors = field_errors(mesh, baseline.values, case.reference, 5)
            integrated = field_errors(mesh, baseline.values, case.reference, 7)
            row = {'case': case.problem.name, 'resolution': n, 'nodes': len(mesh.nodes),
                   'triangles': len(mesh.triangles), 'relative_residual': baseline.relative_residual,
                   'errors': errors,
                   'quadrature_nodal_difference': float(np.max(np.abs(baseline.values-higher.values))),
                   'error_integration_difference': max(abs(errors[k]-integrated[k]) for k in errors)}
            rows.append(row)
            if previous:
                rates.append({'case': case.problem.name, 'coarse': previous['resolution'], 'fine': n,
                    'l2': float(np.log(previous['errors']['absolute_l2']/errors['absolute_l2'])/np.log(n/previous['resolution'])),
                    'h1': float(np.log(previous['errors']['absolute_h1_seminorm']/errors['absolute_h1_seminorm'])/np.log(n/previous['resolution']))})
            previous = row
    passed = (all(r['l2'] >= config['minimum_l2_rate'] and r['h1'] >= config['minimum_h1_rate'] for r in rates)
              and all(r['relative_residual'] < 1e-10
                      and r['quadrature_nodal_difference'] <= config['max_quadrature_nodal_difference']
                      and r['error_integration_difference'] <= config['max_error_integration_difference'] for r in rows))
    return write_manifest(dest/'manifest.json', config, {
        'status': 'passed' if passed else 'failed', 'samples': rows, 'rates': rates}, perf_counter()-started)
