"""Closed symmetric NACA-style profiles and conforming radial exterior grids.

The standard four-digit thickness polynomial is modified only in its x**4
coefficient (-0.1036), enforcing zero thickness at x=1. See docs/airfoil.md.
"""
import numpy as np

from physgraph.core.types import Mesh2D


def _validate_profile(thickness: float, chord: float) -> None:
    if (not np.isfinite([thickness, chord]).all()
            or not .06 <= thickness <= .24 or chord <= 0):
        raise ValueError("require finite 0.06 <= thickness <= 0.24 and chord > 0")


def half_thickness(x: np.ndarray, thickness: float = .12) -> np.ndarray:
    """Dimensionless half thickness at chord fractions x in [0,1]."""
    _validate_profile(thickness, 1.)
    x = np.asarray(x, dtype=float)
    if not np.isfinite(x).all() or np.any((x < 0) | (x > 1)):
        raise ValueError("chord fractions must be finite and in [0, 1]")
    y = 5*thickness*(.2969*np.sqrt(x) + x*(-.1260 + x*(-.3516 + x*(.2843-.1036*x))))
    # Avoid roundoff at the analytically closed leading/trailing endpoints.
    return np.where((x == 0) | (x == 1), 0., y)


def profile_area(thickness: float = .12, chord: float = 1.) -> float:
    """Exact integral 2*c**2*int_0^1 half_thickness(x) dx."""
    _validate_profile(thickness, chord)
    return float(10*thickness*chord**2*(.2969*2/3-.1260/2-.3516/3+.2843/4-.1036/5))


def airfoil_mesh(nr: int, ntheta: int, *, thickness: float = .12,
                 chord: float = 1., farfield_radius: float = 2.,
                 center=(0., 0.), rotation_degrees: float = 0.,
                 grading: float = 1., angular_bias: float = 1.) -> Mesh2D:
    """Triangulate between an airfoil and a concentric polygonal far field.

    center is the chord midpoint. With angular_bias=1, angles are uniform in
    the body frame, with leading and trailing vertices included exactly. A bias < 1 clusters
    rays toward those endpoints using normalized (cos(theta), bias*sin(theta)).
    Each ray intersects the
    convex symmetric profile once; bisection finds that intersection. Layers
    interpolate along the rays, so sectors cannot overlap. Facets are straight.
    """
    for name, count, minimum in (("nr", nr, 1), ("ntheta", ntheta, 16)):
        if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if ntheta % 4:
        raise ValueError("ntheta must be divisible by 4 to retain the profile endpoints")
    _validate_profile(thickness, chord)
    origin = np.asarray(center, dtype=float)
    if origin.shape != (2,) or not np.isfinite(origin).all():
        raise ValueError("center must contain two finite coordinates")
    if (not np.isfinite([farfield_radius, rotation_degrees, grading, angular_bias]).all()
            or farfield_radius <= chord/2 or grading <= 0 or not 0 < angular_bias <= 1):
        raise ValueError("require finite farfield_radius > chord/2, finite rotation, grading > 0 and 0 < angular_bias <= 1")
    theta = 2*np.pi*np.arange(ntheta)/ntheta
    directions = np.column_stack((np.cos(theta), angular_bias*np.sin(theta)))
    if angular_bias != 1.:
        directions /= np.linalg.norm(directions, axis=1)[:, None]
    directions[::ntheta//4] = [[1., 0.], [0., 1.], [-1., 0.], [0., -1.]]
    lo, hi = np.zeros(ntheta), np.full(ntheta, .5)
    for _ in range(60):
        mid = (lo+hi)/2
        x = .5+mid*directions[:, 0]
        inside = mid*np.abs(directions[:, 1]) <= half_thickness(x, thickness)
        lo = np.where(inside, mid, lo)
        hi = np.where(inside, hi, mid)
    radii = chord*(lo+hi)/2
    radii[[0, ntheta//2]] = chord/2
    fraction = np.linspace(0, 1, nr+1)**grading
    layers = radii[None, :] + fraction[:, None]*(farfield_radius-radii)[None, :]
    angle = np.deg2rad(rotation_degrees % 360.)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    nodes = (layers[:, :, None]*directions[None, :, :]).reshape(-1, 2) @ rotation.T + origin
    grid = np.arange((nr+1)*ntheta).reshape(nr+1, ntheta)
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
    tags = np.array(["airfoil_upper"]*(ntheta//2) + ["airfoil_lower"]*(ntheta//2)
                    + ["farfield"]*ntheta)
    return Mesh2D(nodes, triangles, boundary, edges.astype(np.int64), tags)
