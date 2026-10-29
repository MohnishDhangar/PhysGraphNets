"""Mesh-aware evaluation of nodal predictions against separate FEM labels."""
import json
import numpy as np

from physgraph.datasets.catalog import resolve_instance
from physgraph.fem.assembly import assemble
from physgraph.fem.geometry import element_geometry
from physgraph.fem.operators import reduce_system


def prediction_metrics(sample, prediction):
    """Exact P1 L2/H1 differences, prescribed boundary error and free-DOF residual.

    Predictions are neither lifted nor repaired here: boundary violations remain
    visible in both the boundary metric and the full assembled equation residual.
    These errors compare two P1 fields, not a network's continuous derivatives.
    """
    values = np.asarray(prediction)
    if values.shape != sample.fem_solution.shape or values.dtype.kind != 'f' or not np.isfinite(values).all():
        raise ValueError('prediction must contain one finite real floating value per node')
    error_l2 = error_h1 = norm_l2 = norm_h1 = 0.
    for tri in sample.mesh.triangles:
        area, gradients = element_geometry(sample.mesh.nodes[tri])
        target = sample.fem_solution[tri]
        delta = values[tri]-target
        error_l2 += area/12*(delta @ delta+delta.sum()**2)
        norm_l2 += area/12*(target @ target+target.sum()**2)
        error_h1 += area*np.sum((delta @ gradients)**2)
        norm_h1 += area*np.sum((target @ gradients)**2)
    payload = json.loads(sample.instance.canonical_json)
    _, case = resolve_instance(payload['family'], {'geometry': payload['geometry'], 'physics': payload['physics']})
    assembly = assemble(sample.mesh, case.problem, 7 if payload['family'] == 'parametric_square' else 4)
    reduced = reduce_system(sample.mesh, case.problem, assembly)
    residual = (assembly.stiffness @ values-assembly.load)[reduced.interior]
    mask = sample.inputs['dirichlet_mask']
    return {'relative_fem_l2': float(np.sqrt(error_l2/max(norm_l2, 1e-24))),
            'relative_fem_h1_seminorm': float(np.sqrt(error_h1/max(norm_h1, 1e-24))),
            'absolute_fem_l2': float(np.sqrt(error_l2)),
            'absolute_fem_h1_seminorm': float(np.sqrt(error_h1)),
            'max_fem_nodal_error': float(np.max(np.abs(values-sample.fem_solution))),
            'max_dirichlet_error': float(np.max(np.abs(values[mask]-sample.inputs['dirichlet_values'][mask]))),
            'relative_equation_residual': float(np.linalg.norm(residual)/max(np.linalg.norm(reduced.rhs), 1e-12))}
