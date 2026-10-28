"""Conforming O-grid between a circular obstacle and an exact rectangle."""
import numpy as np

from physgraph.core.types import Mesh2D


def cylinder_mesh(nr: int, nside: int, radius: float = .3,
                  bounds=(-1., 1., -.75, .75), center=(0., 0.),
                  grading: float = 1.) -> Mesh2D:
    """Use nside segments per rectangle side and nr radial layers.

    Outer corners are retained exactly. The obstacle is an inscribed polygon;
    rays connect its vertices to the rectangle perimeter, sharing the seam.
    """
    for name, value in (("nr", nr), ("nside", nside)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    box, origin = np.asarray(bounds, dtype=float), np.asarray(center, dtype=float)
    if box.shape != (4,) or origin.shape != (2,) or not np.isfinite(box).all() or not np.isfinite(origin).all():
        raise ValueError("bounds and center must contain four and two finite coordinates")
    xmin, xmax, ymin, ymax = box
    if (not np.isfinite([radius, grading]).all() or radius <= 0 or grading <= 0
            or min(origin[0]-xmin, xmax-origin[0], origin[1]-ymin, ymax-origin[1]) <= radius):
        raise ValueError("circle must lie strictly inside the rectangle; radius and grading must be positive")
    corners = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
    t = np.arange(nside)/nside
    outer = np.concatenate([a+(b-a)*t[:, None] for a, b in zip(corners, np.roll(corners, -1, axis=0))])
    directions = outer-origin
    inner = origin + radius*directions/np.linalg.norm(directions, axis=1)[:, None]
    fraction = np.linspace(0, 1, nr+1)**grading
    nodes = (inner[None, :, :] + fraction[:, None, None]*(outer-inner)[None, :, :]).reshape(-1, 2)
    nt = 4*nside
    grid = np.arange((nr+1)*nt).reshape(nr+1, nt)
    a, b = grid[:-1], grid[1:]
    an, bn = np.roll(a, -1, axis=1), np.roll(b, -1, axis=1)
    triangles = np.stack((np.stack((a, b, bn), -1), np.stack((a, bn, an), -1)), 2).reshape(-1, 3).astype(np.int64)
    v = nodes[triangles]
    e, f = v[:, 1]-v[:, 0], v[:, 2]-v[:, 0]
    signed = e[:, 0]*f[:, 1]-e[:, 1]*f[:, 0]
    if not np.isfinite(signed).all() or np.any(signed <= 0):
        raise ValueError("coordinates produce degenerate or inverted triangles")
    boundary = np.zeros(len(nodes), dtype=bool)
    boundary[grid[[0, -1]].ravel()] = True
    edges = np.vstack((np.column_stack((np.roll(grid[0], -1), grid[0])),
                       np.column_stack((grid[-1], np.roll(grid[-1], -1)))))
    tags = np.array(["cylinder"]*nt + [tag for tag in ("bottom", "right", "top", "left") for _ in range(nside)])
    return Mesh2D(nodes, triangles, boundary, edges.astype(np.int64), tags)
