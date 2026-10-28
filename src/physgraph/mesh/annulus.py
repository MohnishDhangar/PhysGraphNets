"""Periodic radial/angular grids triangulating a polygonal annulus."""
import numpy as np

from physgraph.core.types import Mesh2D


def annulus_mesh(nr: int, ntheta: int, inner_radius: float = .5,
                 outer_radius: float = 1., center: tuple[float, float] = (0., 0.),
                 grading: float = 1.) -> Mesh2D:
    """Split each polar cell into two CCW P1 triangles, with a shared seam.

    nr counts radial cells; ntheta counts angular sectors. Boundary vertices
    lie on the circles, but facets are straight chords. Tags are 'inner' and
    'outer'; oriented inner edges run clockwise and outer edges run CCW.
    """
    for name, n, minimum in (("nr", nr, 1), ("ntheta", ntheta, 3)):
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if (not np.isfinite([inner_radius, outer_radius, grading]).all()
            or not 0 < inner_radius < outer_radius or grading <= 0):
        raise ValueError("require finite 0 < inner_radius < outer_radius and grading > 0")
    origin = np.asarray(center, dtype=float)
    if origin.shape != (2,) or not np.isfinite(origin).all():
        raise ValueError("center must contain two finite coordinates")
    radii = inner_radius + (outer_radius-inner_radius)*np.linspace(0, 1, nr+1)**grading
    if np.any(np.diff(radii) <= 0):
        raise ValueError("radial spacing is too small to represent at this precision")
    theta = 2*np.pi*np.arange(ntheta)/ntheta
    directions = np.column_stack((np.cos(theta), np.sin(theta)))
    nodes = (radii[:, None, None]*directions[None, :, :] + origin).reshape(-1, 2)
    grid = np.arange((nr+1)*ntheta).reshape(nr+1, ntheta)
    inner, outer = grid[:-1], grid[1:]
    inner_next, outer_next = np.roll(inner, -1, axis=1), np.roll(outer, -1, axis=1)
    triangles = np.stack((np.stack((inner, outer, outer_next), axis=-1),
                          np.stack((inner, outer_next, inner_next), axis=-1)),
                         axis=2).reshape(-1, 3).astype(np.int64)
    # Detect representational collapse after translation as well as radial collapse.
    vertices = nodes[triangles]
    first, second = vertices[:, 1]-vertices[:, 0], vertices[:, 2]-vertices[:, 0]
    signed = first[:, 0]*second[:, 1] - first[:, 1]*second[:, 0]
    if not np.isfinite(signed).all() or np.any(signed <= 0):
        raise ValueError("annulus coordinates produce degenerate or inverted triangles")
    boundary = np.zeros(len(nodes), dtype=bool)
    boundary[grid[[0, -1]].ravel()] = True
    inner_edges = np.column_stack((np.roll(grid[0], -1), grid[0]))
    outer_edges = np.column_stack((grid[-1], np.roll(grid[-1], -1)))
    edges = np.vstack((inner_edges, outer_edges)).astype(np.int64)
    tags = np.array(["inner"]*ntheta + ["outer"]*ntheta)
    return Mesh2D(nodes, triangles, boundary, edges, tags)
