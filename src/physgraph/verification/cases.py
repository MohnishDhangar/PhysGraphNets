"""Small reproducible fixtures shared with independent solver adapters."""
from dataclasses import dataclass
from physgraph.physics.mixed_square import mixed_case, CASE_NAMES as MIXED_CASE_NAMES

from physgraph.core.types import Mesh2D
from physgraph.physics.diffusion import ManufacturedEllipticCase, get_elliptic_case
from physgraph.mesh.nonuniform import graded_square_mesh
from physgraph.mesh.annulus import annulus_mesh
from physgraph.mesh.cylinder import cylinder_mesh
from physgraph.mesh.airfoil import airfoil_mesh
from physgraph.physics.annulus import radial_log_case
from physgraph.physics.cylinder import cylinder_case
from physgraph.physics.airfoil import airfoil_case
from physgraph.physics.parametric_square import square_case

PARAMETRIC_CASE_NAMES = ('parametric_square_dirichlet', 'parametric_square_right_neumann')

CASE_IDS = ('variable_sine', 'affine_diffusion', 'mixed_polynomial',
            'annulus_dirichlet', 'annulus_inner_neumann', 'annulus_outer_neumann',
            'cylinder_dirichlet', 'cylinder_cylinder_neumann', 'cylinder_outer_neumann',
            'airfoil_dirichlet', 'airfoil_airfoil_neumann', 'airfoil_farfield_neumann') + MIXED_CASE_NAMES + PARAMETRIC_CASE_NAMES


@dataclass(frozen=True)
class ComparisonCase:
    name: str
    mesh: Mesh2D
    manufactured: ManufacturedEllipticCase
    dirichlet_tags: tuple[str, ...] | None


def comparison_case(name: str, resolution: int = 4) -> ComparisonCase:
    if name not in CASE_IDS:
        raise ValueError(f'unknown comparison case: {name}')
    if isinstance(resolution, bool) or not isinstance(resolution, int) or resolution < 2:
        raise ValueError('resolution must be an integer >= 2')
    n = resolution
    if name in CASE_IDS[:3]:
        return ComparisonCase(name, graded_square_mesh(n, grading=1.3), get_elliptic_case(name), None)
    if name in MIXED_CASE_NAMES:
        return ComparisonCase(name, graded_square_mesh(n, grading=1.3), mixed_case(name), None)
    if name in PARAMETRIC_CASE_NAMES:
        return ComparisonCase(name, graded_square_mesh(n, grading=1.3),
                              square_case(boundary_mode=name.removeprefix('parametric_square_')), None)
    geometry, mode = name.split('_', 1)
    if geometry == 'annulus':
        mesh, case = annulus_mesh(n, 8*n), radial_log_case(boundary_mode=mode)
        tags = {'dirichlet': ('inner', 'outer'), 'inner_neumann': ('outer',), 'outer_neumann': ('inner',)}[mode]
    elif geometry == 'cylinder':
        mesh, case = cylinder_mesh(n, 2*n), cylinder_case(boundary_mode=mode)
        outer = ('bottom', 'right', 'top', 'left')
        tags = {'dirichlet': ('cylinder',)+outer, 'cylinder_neumann': outer, 'outer_neumann': ('cylinder',)}[mode]
    else:
        mesh, case = airfoil_mesh(n, 16*n), airfoil_case(boundary_mode=mode)
        inner = ('airfoil_upper', 'airfoil_lower')
        tags = {'dirichlet': inner+('farfield',), 'airfoil_neumann': ('farfield',), 'farfield_neumann': inner}[mode]
    return ComparisonCase(name, mesh, case, tags)
