"""Manufactured 2D Poisson cases with known analytical solutions."""
import numpy as np
from numpy.typing import NDArray

from physgraph.core.types import ManufacturedCase


def _sine(p: NDArray[np.float64]) -> NDArray[np.float64]:
    x, y = p[..., 0], p[..., 1]
    return np.sin(np.pi * x) * np.sin(np.pi * y)


def _sine_gradient(p: NDArray[np.float64]) -> NDArray[np.float64]:
    x, y = p[..., 0], p[..., 1]
    return np.stack((np.pi * np.cos(np.pi * x) * np.sin(np.pi * y),
                     np.pi * np.sin(np.pi * x) * np.cos(np.pi * y)), axis=-1)


def _polynomial(p: NDArray[np.float64]) -> NDArray[np.float64]:
    x, y = p[..., 0], p[..., 1]
    return x * (1 - x) * y * (1 - y)


def _poly_force(p: NDArray[np.float64]) -> NDArray[np.float64]:
    x, y = p[..., 0], p[..., 1]
    return 2 * (x * (1 - x) + y * (1 - y))


def _poly_gradient(p: NDArray[np.float64]) -> NDArray[np.float64]:
    x, y = p[..., 0], p[..., 1]
    return np.stack(((1 - 2 * x) * y * (1 - y),
                     x * (1 - x) * (1 - 2 * y)), axis=-1)


def _affine(p: NDArray[np.float64]) -> NDArray[np.float64]:
    return 1 + 2 * p[..., 0] - 3 * p[..., 1]


def get_case(name: str) -> ManufacturedCase:
    """Return known solutions, including nonzero Dirichlet test 'affine'."""
    if name == "sine":
        return ManufacturedCase(name, _sine,
                                lambda p: 2 * np.pi**2 * _sine(p), _sine_gradient)
    if name == "polynomial":
        return ManufacturedCase(name, _polynomial, _poly_force, _poly_gradient)
    if name == "affine":
        return ManufacturedCase(name, _affine,
                                lambda p: np.zeros(p.shape[:-1], dtype=float),
                                lambda p: np.broadcast_to([2., -3.], p.shape))
    raise ValueError(f"Unknown case {name!r}. Choose sine, polynomial, or affine.")
