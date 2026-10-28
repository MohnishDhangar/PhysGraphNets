"""Physical inputs and optional analytical references are separate contracts."""
from dataclasses import dataclass

import numpy as np

from physgraph.core.types import ScalarFunction, VectorFunction
from physgraph.physics.boundary import BoundaryConditions


@dataclass(frozen=True)
class EllipticProblem:
    """-div(k grad u) + c u = f; scalar k > 0, c >= 0."""

    name: str
    diffusion: ScalarFunction
    reaction: ScalarFunction
    forcing: ScalarFunction
    boundary: BoundaryConditions


@dataclass(frozen=True)
class AnalyticalReference:
    exact: ScalarFunction
    gradient: VectorFunction


def sample_scalar(function: ScalarFunction, points: np.ndarray, name: str) -> np.ndarray:
    """Require one finite scalar per quadrature/node point; no broadcasting."""
    values = np.asarray(function(points), dtype=float)
    if values.shape != (len(points),) or not np.isfinite(values).all():
        raise ValueError(f"{name} must return one finite scalar per point")
    return values
