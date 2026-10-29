"""Uniform P1 triangular meshes on the unit square (Phase 1 only)."""
import numpy as np

from physgraph.core.types import Mesh2D


def unit_square_mesh(nx: int, ny: int | None = None) -> Mesh2D:
    """Split each structured quad into two CCW triangles."""
    if ny is None:
        ny = nx
    for n in (nx, ny):
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 2:
            raise ValueError("nx and ny must be integers >= 2")

    xx, yy = np.meshgrid(np.linspace(0, 1, nx + 1),
                         np.linspace(0, 1, ny + 1), indexing="xy")
    nodes = np.column_stack((xx.ravel(), yy.ravel())).astype(np.float64)
    boundary = ((np.isclose(nodes[:, 0], 0) | np.isclose(nodes[:, 0], 1)) |
                (np.isclose(nodes[:, 1], 0) | np.isclose(nodes[:, 1], 1)))

    triangles: list[tuple[int, int, int]] = []
    stride = nx + 1
    for j in range(ny):
        for i in range(nx):
            ll = j * stride + i
            lr = ll + 1
            ul = ll + stride
            ur = ul + 1
            triangles.extend(((ll, lr, ur), (ll, ur, ul)))
    return Mesh2D(nodes=nodes, triangles=np.asarray(triangles, dtype=np.int64),
                  boundary=boundary)


def tagged_unit_square_mesh(nx: int, ny: int | None = None) -> Mesh2D:
    """The same square triangulation with every side explicitly tagged."""
    mesh = unit_square_mesh(nx, ny)
    stride = (nx + 1)
    ny = nx if ny is None else ny
    bottom = np.arange(nx)
    right = np.arange(ny)*stride + nx
    top = ny*stride + np.arange(nx, 0, -1)
    left = np.arange(ny, 0, -1)*stride
    edges = np.vstack((np.column_stack((bottom, bottom+1)),
                       np.column_stack((right, right+stride)),
                       np.column_stack((top, top-1)),
                       np.column_stack((left, left-stride))))
    tags = np.array(['bottom']*nx + ['right']*ny + ['top']*nx + ['left']*ny)
    return Mesh2D(mesh.nodes, mesh.triangles, mesh.boundary, edges, tags)
