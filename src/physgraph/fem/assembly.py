"""P1 weak-form assembly for scalar diffusion/reaction with mixed boundaries."""
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from physgraph.core.types import Mesh2D
from physgraph.fem.boundary import BoundaryPartition, partition_boundary
from physgraph.fem.geometry import element_geometry
from physgraph.fem.quadrature import interval_rule, triangle_rule
from physgraph.physics.base import EllipticProblem, sample_scalar


@dataclass(frozen=True)
class EllipticAssembly:
    stiffness: csr_matrix  # Includes both diffusion and reaction.
    mass: csr_matrix
    load: np.ndarray
    boundary: BoundaryPartition


def assemble(mesh: Mesh2D, problem: EllipticProblem,
             quadrature_order: int = 4) -> EllipticAssembly:
    """Assemble A_ij=int(k grad phi_i.grad phi_j+c phi_i phi_j), M, F.

    Coefficient admissibility is checked at integration points; callers remain
    responsible for positivity throughout each element for arbitrary functions.
    """
    bary, weights = triangle_rule(quadrature_order)
    if quadrature_order < 2:
        raise ValueError("assembly quadrature order must be at least 2 for the P1 mass matrix")
    partition = partition_boundary(mesh, problem.boundary)
    count = len(mesh.nodes)
    local_matrices = np.empty((len(mesh.triangles), 3, 3))
    local_masses = np.empty_like(local_matrices)
    rhs = np.zeros(count)
    reference_mass = (bary.T * weights) @ bary
    for index, tri in enumerate(mesh.triangles):
        vertices = mesh.nodes[tri]
        area, gradients = element_geometry(vertices)
        points = bary @ vertices
        k = sample_scalar(problem.diffusion, points, "diffusion")
        c = sample_scalar(problem.reaction, points, "reaction")
        f = sample_scalar(problem.forcing, points, "forcing")
        if np.any(k <= 0) or np.any(c < 0):
            raise ValueError("diffusion must be positive and reaction nonnegative at quadrature points")
        local_matrices[index] = area * (
            (weights @ k) * (gradients @ gradients.T)
            + (bary.T * (weights*c)) @ bary)
        local_masses[index] = area * reference_mass
        rhs[tri] += area * (bary.T @ (weights*f))

    t, edge_weights = interval_rule(quadrature_order)
    shape = np.column_stack((1-t, t))
    for edge in partition.neumann_edges:
        vertices = mesh.nodes[edge]
        tangent = vertices[1]-vertices[0]
        length = np.linalg.norm(tangent)
        normal = np.array([tangent[1], -tangent[0]]) / length
        points = shape @ vertices
        normals = np.broadcast_to(normal, points.shape)
        q = np.asarray(problem.boundary.neumann(points, normals), dtype=float)
        if q.shape != (len(points),) or not np.isfinite(q).all():
            raise ValueError("Neumann data must return one finite scalar per point")
        rhs[edge] += length * (shape.T @ (edge_weights*q))

    rows = np.repeat(mesh.triangles, 3, axis=1).ravel()
    cols = np.tile(mesh.triangles, (1, 3)).ravel()
    matrix = coo_matrix((local_matrices.ravel(), (rows, cols)), shape=(count, count)).tocsr()
    mass = coo_matrix((local_masses.ravel(), (rows, cols)), shape=(count, count)).tocsr()
    if not all(np.isfinite(a).all() for a in (matrix.data, mass.data, rhs)):
        raise ValueError("assembly produced nonfinite values")
    return EllipticAssembly(matrix, mass, rhs, partition)
