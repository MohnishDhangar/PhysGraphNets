"""Manufactured diffusion/reaction fields on a rectangle with a circular hole."""
import numpy as np
from physgraph.physics.base import AnalyticalReference, EllipticProblem
from physgraph.physics.boundary import BoundaryConditions
from physgraph.physics.diffusion import ManufacturedEllipticCase

BOUNDARY_MODES = ("dirichlet", "cylinder_neumann", "outer_neumann")
CASES = ("quadratic", "affine")


def cylinder_case(case="quadratic", boundary_mode="dirichlet", diffusivity=1.):
    """Solve -k Δu + u=f; prescribe q=k grad(u)·n on actual facets."""
    if case not in CASES or boundary_mode not in BOUNDARY_MODES:
        raise ValueError("unknown cylinder case or boundary mode")
    if not np.isfinite(diffusivity) or diffusivity <= 0:
        raise ValueError("diffusivity must be finite and positive")
    def exact(p):
        return 1+p[:, 0]**2+p[:, 1]**2 if case == "quadratic" else 1+2*p[:, 0]-3*p[:, 1]
    def gradient(p):
        return 2*p if case == "quadratic" else np.tile([2., -3.], (len(p), 1))
    def forcing(p):
        return exact(p)-(4*diffusivity if case == "quadratic" else 0.)
    outer = ("bottom", "right", "top", "left")
    tags = {"dirichlet": ("cylinder",)+outer, "cylinder_neumann": outer,
            "outer_neumann": ("cylinder",)}[boundary_mode]
    boundary = BoundaryConditions(exact, dirichlet_tags=tags,
        neumann=None if boundary_mode == "dirichlet" else lambda p, n: diffusivity*np.sum(gradient(p)*n, axis=1))
    problem = EllipticProblem(f"cylinder_{case}_{boundary_mode}",
        lambda p: np.full(len(p), diffusivity), lambda p: np.ones(len(p)), forcing, boundary)
    return ManufacturedEllipticCase(problem, AnalyticalReference(exact, gradient))
