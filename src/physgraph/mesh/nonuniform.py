"""Nonuniform, conforming triangular meshes without external meshing tools."""
import numpy as np

from physgraph.core.types import Mesh2D


def tensor_square_mesh(x: np.ndarray, y: np.ndarray) -> Mesh2D:
    """Triangulate an increasing tensor grid spanning the unit square."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    for axis in (x, y):
        if (axis.ndim != 1 or len(axis) < 3 or not np.isfinite(axis).all()
                or axis[0] != 0 or axis[-1] != 1 or np.any(np.diff(axis) <= 0)):
            raise ValueError("axes must strictly increase from 0 to 1 with at least 3 points")
    xx, yy = np.meshgrid(x, y, indexing="xy")
    nodes = np.column_stack((xx.ravel(), yy.ravel()))
    nx, ny = len(x)-1, len(y)-1
    grid = np.arange(len(nodes)).reshape(ny+1, nx+1)
    ll, lr = grid[:-1, :-1].ravel(), grid[:-1, 1:].ravel()
    ul, ur = grid[1:, :-1].ravel(), grid[1:, 1:].ravel()
    triangles = np.stack((np.column_stack((ll, lr, ur)),
                          np.column_stack((ll, ur, ul))), axis=1).reshape(-1, 3)
    boundary = np.zeros((ny+1, nx+1), dtype=bool)
    boundary[[0, -1], :] = True
    boundary[:, [0, -1]] = True
    return Mesh2D(nodes, triangles.astype(np.int64), boundary.ravel())


def graded_square_mesh(nx: int, ny: int | None = None, grading: float = 1.3) -> Mesh2D:
    """Use x_i=(i/nx)^grading and y_j=(j/ny)^grading, preserving CCW cells."""
    ny = nx if ny is None else ny
    for n in (nx, ny):
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 2:
            raise ValueError("nx and ny must be integers >= 2")
    if not np.isfinite(grading) or grading <= 0:
        raise ValueError("grading must be positive and finite")
    return tensor_square_mesh(np.linspace(0, 1, nx+1)**grading,
                              np.linspace(0, 1, ny+1)**grading)
