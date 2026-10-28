"""Quadrature-based L2 and H1 seminorm errors against an exact field."""
import numpy as np
from physgraph.core.types import ManufacturedCase, Mesh2D
from physgraph.fem.geometry import element_geometry
from physgraph.fem.quadrature import triangle_rule
from physgraph.physics.base import AnalyticalReference, sample_scalar


def field_errors(mesh: Mesh2D, values: np.ndarray,
                 case: ManufacturedCase | AnalyticalReference,
                 quadrature_order: int = 5) -> dict[str, float]:
    if values.shape != (len(mesh.nodes),) or not np.isfinite(values).all():
        raise ValueError("values must be finite and have shape (N,)")
    barycentric, weights = triangle_rule(quadrature_order)
    e_l2 = e_h1 = norm_l2 = norm_h1 = 0.0
    for tri in mesh.triangles:
        vertices = mesh.nodes[tri]
        area, gradients = element_geometry(vertices)
        points = barycentric @ vertices
        uh = barycentric @ values[tri]
        ue = sample_scalar(case.exact, points, "analytical solution")
        gh = values[tri] @ gradients
        ge = np.asarray(case.gradient(points))
        if ge.shape != points.shape or not np.isfinite(ge).all():
            raise ValueError("analytical gradient must return one finite 2-vector per point")
        e_l2 += area * float(weights @ (uh - ue)**2)
        e_h1 += area * float(weights @ np.sum((ge - gh)**2, axis=1))
        norm_l2 += area * float(weights @ ue**2)
        norm_h1 += area * float(weights @ np.sum(ge**2, axis=1))
    nodal_error = values - sample_scalar(case.exact, mesh.nodes, "analytical solution")
    return {
        "relative_l2": float(np.sqrt(e_l2 / max(norm_l2, 1e-24))),
        "relative_h1_seminorm": float(np.sqrt(e_h1 / max(norm_h1, 1e-24))),
        "max_nodal_error": float(np.max(np.abs(nodal_error))),
        "absolute_l2": float(np.sqrt(e_l2)),
        "absolute_h1_seminorm": float(np.sqrt(e_h1)),
    }
