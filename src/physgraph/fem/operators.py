"""Reusable reduced operator for reference solves and future physics losses."""
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from physgraph.core.types import Mesh2D
from physgraph.fem.assembly import EllipticAssembly
from physgraph.physics.base import EllipticProblem, sample_scalar


@dataclass(frozen=True)
class ReducedSystem:
    matrix: csr_matrix
    rhs: np.ndarray
    interior: np.ndarray  # Free DOFs, including nodes on Neumann boundaries.
    boundary: np.ndarray  # Prescribed DOFs only.
    boundary_values: np.ndarray
    node_count: int

    def _validate(self, free_values: np.ndarray) -> np.ndarray:
        values = np.asarray(free_values, dtype=float)
        if values.shape != self.rhs.shape or not np.isfinite(values).all():
            raise ValueError("free values must be finite and match the reduced system")
        return values

    def lift(self, free_values: np.ndarray) -> np.ndarray:
        values = np.empty(self.node_count)
        values[self.interior] = self._validate(free_values)
        values[self.boundary] = self.boundary_values
        return values

    def residual(self, free_values: np.ndarray) -> np.ndarray:
        return self.matrix @ self._validate(free_values) - self.rhs

    def energy(self, free_values: np.ndarray) -> float:
        values = self._validate(free_values)
        return float(0.5 * values @ (self.matrix @ values) - values @ self.rhs)


def reduce_system(mesh: Mesh2D, problem: EllipticProblem,
                  assembly: EllipticAssembly) -> ReducedSystem:
    boundary = assembly.boundary.dirichlet_nodes
    free = np.ones(len(mesh.nodes), dtype=bool)
    free[boundary] = False
    interior = np.flatnonzero(free)
    g = sample_scalar(problem.boundary.dirichlet, mesh.nodes[boundary], "Dirichlet data")
    count, labels = connected_components(assembly.stiffness, directed=False)
    if len(np.unique(labels[boundary])) != count:
        raise ValueError("each connected mesh component must have a Dirichlet boundary")
    matrix = assembly.stiffness[interior][:, interior].tocsr()
    rhs = assembly.load[interior] - assembly.stiffness[interior][:, boundary] @ g
    return ReducedSystem(matrix, rhs, interior, boundary, g, len(mesh.nodes))
