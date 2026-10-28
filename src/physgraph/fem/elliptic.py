"""CPU reference solver; no analytical solution is required by this API."""
from dataclasses import dataclass
import warnings

import numpy as np
from scipy.sparse.linalg import MatrixRankWarning, spsolve

from physgraph.core.types import Mesh2D
from physgraph.fem.assembly import EllipticAssembly, assemble
from physgraph.fem.operators import ReducedSystem, reduce_system
from physgraph.physics.base import EllipticProblem


@dataclass(frozen=True)
class EllipticSolution:
    values: np.ndarray
    assembly: EllipticAssembly
    reduced: ReducedSystem
    absolute_residual: float
    relative_residual: float


def solve(mesh: Mesh2D, problem: EllipticProblem,
          quadrature_order: int = 4) -> EllipticSolution:
    assembly = assemble(mesh, problem, quadrature_order)
    reduced = reduce_system(mesh, problem, assembly)
    with warnings.catch_warnings():
        warnings.simplefilter("error", MatrixRankWarning)
        free = spsolve(reduced.matrix, reduced.rhs) if len(reduced.interior) else np.empty(0)
    values = reduced.lift(free)
    absolute = float(np.linalg.norm(reduced.residual(free)))
    relative = absolute / max(float(np.linalg.norm(reduced.rhs)), 1e-12)
    return EllipticSolution(values, assembly, reduced, absolute, relative)
