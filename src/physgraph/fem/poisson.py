"""P1 triangular finite element assembly for -Δu = f and Dirichlet data."""
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

from physgraph.core.types import ManufacturedCase, Mesh2D
from physgraph.fem.geometry import element_geometry
from physgraph.physics.base import sample_scalar

# Three-point symmetric reference triangle rule; weights sum to 1.
BARYCENTRIC = np.array([[2/3, 1/6, 1/6],
                        [1/6, 2/3, 1/6],
                        [1/6, 1/6, 2/3]], dtype=np.float64)


@dataclass(frozen=True)
class FEMSolution:
    values: NDArray[np.float64]
    stiffness: csr_matrix
    load: NDArray[np.float64]
    relative_residual: float


def assemble(mesh: Mesh2D, case: ManufacturedCase) -> tuple[csr_matrix, NDArray[np.float64]]:
    """Assemble sparse K and F using exact P1 stiffness and 3-point force quadrature."""
    count = len(mesh.nodes)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    rhs = np.zeros(count, dtype=np.float64)

    for tri in mesh.triangles:
        verts = mesh.nodes[tri]
        area, grads = element_geometry(verts)
        local_k = area * (grads @ grads.T)
        qpts = BARYCENTRIC @ verts
        f_q = sample_scalar(case.forcing, qpts, "forcing")
        local_f = area / 3.0 * (BARYCENTRIC.T @ f_q)
        rhs[tri] += local_f
        for i in range(3):
            for j in range(3):
                rows.append(int(tri[i]))
                cols.append(int(tri[j]))
                vals.append(float(local_k[i, j]))

    matrix = coo_matrix((vals, (rows, cols)), shape=(count, count)).tocsr()
    matrix.sum_duplicates()
    return matrix, rhs


def solve(mesh: Mesh2D, case: ManufacturedCase) -> FEMSolution:
    """Solve reduced Dirichlet system; never penalize boundary equations."""
    k, rhs = assemble(mesh, case)
    interior = np.flatnonzero(~mesh.boundary)
    boundary = np.flatnonzero(mesh.boundary)
    if len(interior) == 0:
        raise ValueError("mesh has no interior degrees of freedom")

    u = np.zeros(len(mesh.nodes), dtype=float)
    u[boundary] = sample_scalar(case.exact, mesh.nodes[boundary], "Dirichlet data")
    kii = k[interior][:, interior].tocsr()
    reduced_rhs = rhs[interior] - k[interior][:, boundary] @ u[boundary]
    u[interior] = spsolve(kii, reduced_rhs)
    if not np.isfinite(u).all():
        raise ValueError("Poisson solve produced nonfinite values")
    residual = (k @ u - rhs)[interior]
    relative = float(np.linalg.norm(residual) / max(np.linalg.norm(reduced_rhs), 1e-12))
    return FEMSolution(values=u, stiffness=k, load=rhs, relative_residual=relative)
