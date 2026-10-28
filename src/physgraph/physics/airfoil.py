"""Manufactured elliptic references on tagged airfoil exterior domains."""
from dataclasses import replace

from physgraph.physics.cylinder import cylinder_case, CASES
from physgraph.physics.diffusion import ManufacturedEllipticCase

BOUNDARY_MODES = ("dirichlet", "airfoil_neumann", "farfield_neumann")


def airfoil_case(case: str = "quadratic", boundary_mode: str = "dirichlet",
                 diffusivity: float = 1.) -> ManufacturedEllipticCase:
    """Reuse geometry-independent polynomial fields with airfoil boundary tags.

    -k Delta u + u=f; q=k grad(u).n uses actual outward facet normals.
    No aerodynamic boundary conditions or flow equations are implied.
    """
    if boundary_mode not in BOUNDARY_MODES:
        raise ValueError(f"boundary_mode must be one of {BOUNDARY_MODES}")
    # Polynomial PDE/reference data are identical on both geometry families.
    base = cylinder_case(case, "dirichlet" if boundary_mode == "dirichlet"
                         else "cylinder_neumann", diffusivity)
    surface = ("airfoil_upper", "airfoil_lower")
    tags = {"dirichlet": surface+("farfield",), "airfoil_neumann": ("farfield",),
            "farfield_neumann": surface}[boundary_mode]
    boundary = replace(base.problem.boundary, dirichlet_tags=tags)
    problem = replace(base.problem, name=f"airfoil_{case}_{boundary_mode}", boundary=boundary)
    return ManufacturedEllipticCase(problem, base.reference)
