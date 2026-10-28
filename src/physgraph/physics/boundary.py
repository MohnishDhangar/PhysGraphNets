"""Boundary data for scalar elliptic equations.

Neumann data is outward diffusive derivative q = k grad(u) dot n, not the
physical flux -k grad(u) dot n. This convention gives + integral(q*v) in F.
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np

from physgraph.core.types import ScalarFunction

BoundarySelector = Callable[[np.ndarray], np.ndarray]
FluxFunction = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class BoundaryConditions:
    """Select Dirichlet edges by named tags or a midpoint predicate.

    Endpoints of selected edges are prescribed, including mixed-boundary
    corners. Every remaining exterior edge receives the Neumann function.
    Mesh edges must align with changes in boundary type. With neither tags
    nor a selector, all exterior edges are Dirichlet.
    """

    dirichlet: ScalarFunction
    dirichlet_selector: BoundarySelector | None = None
    neumann: FluxFunction | None = None
    dirichlet_tags: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.dirichlet_tags is not None:
            if self.dirichlet_selector is not None:
                raise ValueError("choose either dirichlet_tags or dirichlet_selector")
            if (not isinstance(self.dirichlet_tags, tuple) or not self.dirichlet_tags
                    or any(not isinstance(tag, str) or not tag.strip() for tag in self.dirichlet_tags)
                    or len(set(self.dirichlet_tags)) != len(self.dirichlet_tags)):
                raise ValueError("dirichlet_tags must be a nonempty tuple of distinct names")
