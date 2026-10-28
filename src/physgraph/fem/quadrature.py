"""Gauss integration on triangles and edges, independent of any PDE."""
from functools import lru_cache

import numpy as np
from numpy.polynomial.legendre import leggauss


@lru_cache(maxsize=16)
def interval_rule(order: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Points and weights on [0, 1]; integrates degree 2*order-1 exactly."""
    if isinstance(order, bool) or not isinstance(order, (int, np.integer)) or order < 1:
        raise ValueError("quadrature order must be a positive integer")
    points, weights = leggauss(order)
    points, weights = (points + 1) / 2, weights / 2
    points.setflags(write=False)
    weights.setflags(write=False)
    return points, weights


@lru_cache(maxsize=16)
def triangle_rule(order: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Duffy product rule: barycentric points and weights summing to one.

    Multiply by physical triangle area. Total polynomial degree <= 2*order-2
    is integrated exactly. Higher order is needed for nonpolynomial inputs.
    """
    t, w = interval_rule(order)
    r, s = np.meshgrid(t, t, indexing="ij")
    barycentric = np.stack(((1-r)*(1-s), r, (1-r)*s), axis=-1).reshape(-1, 3)
    weights = (2 * w[:, None] * w[None, :] * (1-r)).ravel()
    barycentric.setflags(write=False)
    weights.setflags(write=False)
    return barycentric, weights
