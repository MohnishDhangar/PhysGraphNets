"""Analytical radial harmonic field for annulus geometry verification."""
import numpy as np

from physgraph.physics.base import AnalyticalReference, EllipticProblem
from physgraph.physics.boundary import BoundaryConditions
from physgraph.physics.diffusion import ManufacturedEllipticCase

BOUNDARY_MODES = ("dirichlet", "inner_neumann", "outer_neumann")


def radial_log_case(inner_radius: float = .5, outer_radius: float = 1.,
                    inner_value: float = 1., outer_value: float = 2.,
                    diffusivity: float = 1., center: tuple[float, float] = (0., 0.),
                    boundary_mode: str = "dirichlet") -> ManufacturedEllipticCase:
    """Solve -k Delta u=0 with u=ui+B log(r/a), B=(uo-ui)/log(b/a).

    Mixed variants prescribe the exact k*grad(u).n on polygon facets, using
    their actual outward normals. The reference is smooth for r>0 and hence
    extends to the small chordal differences from the ideal circular domain.
    """
    if (not np.isfinite([inner_radius, outer_radius, inner_value, outer_value, diffusivity]).all()
            or not 0 < inner_radius < outer_radius or diffusivity <= 0):
        raise ValueError("require finite values, 0 < inner_radius < outer_radius, and diffusivity > 0")
    origin = np.asarray(center, dtype=float)
    if origin.shape != (2,) or not np.isfinite(origin).all():
        raise ValueError("center must contain two finite coordinates")
    if boundary_mode not in BOUNDARY_MODES:
        raise ValueError(f"boundary_mode must be one of {BOUNDARY_MODES}")
    scale = (outer_value-inner_value)/np.log(outer_radius/inner_radius)

    def relative(points):
        delta = np.asarray(points, dtype=float)-origin
        r2 = np.sum(delta*delta, axis=1)
        if not np.isfinite(r2).all() or np.any(r2 <= 0):
            raise ValueError("radial reference requires finite points away from the center")
        return delta, r2

    def exact(points):
        _, r2 = relative(points)
        return inner_value + scale*np.log(np.sqrt(r2)/inner_radius)

    def gradient(points):
        delta, r2 = relative(points)
        return scale*delta/r2[:, None]

    def neumann(points, normals):
        return diffusivity*np.sum(gradient(points)*normals, axis=1)

    prescribed = {"dirichlet": ("inner", "outer"),
                  "inner_neumann": ("outer",), "outer_neumann": ("inner",)}[boundary_mode]
    boundary = BoundaryConditions(exact, neumann=neumann if boundary_mode != "dirichlet" else None,
                                   dirichlet_tags=prescribed)
    problem = EllipticProblem(f"annulus_log_{boundary_mode}",
                              lambda p: np.full(len(p), diffusivity),
                              lambda p: np.zeros(len(p)), lambda p: np.zeros(len(p)), boundary)
    return ManufacturedEllipticCase(problem, AnalyticalReference(exact, gradient))
