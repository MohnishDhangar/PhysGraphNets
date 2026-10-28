from dataclasses import replace
import math

import numpy as np
import pytest

from physgraph.core.types import Mesh2D
from physgraph.experiments.elliptic import evaluate
from physgraph.fem.assembly import assemble
from physgraph.fem.elliptic import solve
from physgraph.fem.error import field_errors
from physgraph.fem.geometry import element_geometry
from physgraph.fem.poisson import assemble as assemble_poisson
from physgraph.fem.quadrature import triangle_rule
from physgraph.mesh.nonuniform import graded_square_mesh, tensor_square_mesh
from physgraph.mesh.structured import unit_square_mesh
from physgraph.physics.base import EllipticProblem
from physgraph.physics.boundary import BoundaryConditions
from physgraph.physics.diffusion import get_elliptic_case
from physgraph.physics.manufactured import get_case


def constant(value):
    return lambda p: np.full(len(p), value, dtype=float)


def test_triangle_quadrature_against_exact_monomial_integrals():
    bary, weights = triangle_rule(4)
    x, y = bary[:, 1], bary[:, 2]
    for degree in range(7):
        for a in range(degree+1):
            b = degree-a
            expected = math.factorial(a)*math.factorial(b)/math.factorial(a+b+2)
            assert 0.5*(weights @ (x**a*y**b)) == pytest.approx(expected, abs=1e-14)


def test_element_stiffness_mass_and_load_against_analytic_matrices():
    mesh = Mesh2D(np.array([[0., 0.], [1., 0.], [0., 1.]]),
                  np.array([[0, 1, 2]]), np.ones(3, dtype=bool))
    problem = EllipticProblem("constant", constant(2), constant(3), constant(4),
                              BoundaryConditions(constant(0)))
    assembled = assemble(mesh, problem)
    stiffness = np.array([[1., -.5, -.5], [-.5, .5, 0], [-.5, 0, .5]])
    mass = (np.ones((3, 3)) + np.eye(3))/24
    np.testing.assert_allclose(assembled.stiffness.toarray(), 2*stiffness+3*mass, atol=1e-14)
    np.testing.assert_allclose(assembled.mass.toarray(), mass, atol=1e-14)
    np.testing.assert_allclose(assembled.load, np.full(3, 2/3), atol=1e-14)


def test_poisson_operator_matches_existing_independent_assembly():
    mesh = unit_square_mesh(5, 6)
    case = get_case("affine")
    problem = EllipticProblem("poisson", constant(1), constant(0), case.forcing,
                              BoundaryConditions(case.exact))
    old_k, old_f = assemble_poisson(mesh, case)
    new = assemble(mesh, problem)
    np.testing.assert_allclose(new.stiffness.toarray(), old_k.toarray(), atol=1e-13)
    np.testing.assert_allclose(new.load, old_f, atol=1e-14)


def test_affine_patch_on_perturbed_nonuniform_mesh():
    original = graded_square_mesh(7, 6, grading=1.3)
    nodes = original.nodes.copy()
    free = ~original.boundary
    nodes[free] += np.random.default_rng(7).uniform(-0.004, 0.004, (free.sum(), 2))
    mesh = Mesh2D(nodes, original.triangles, original.boundary)
    case = get_elliptic_case("affine_diffusion")
    result = solve(mesh, case.problem)
    np.testing.assert_allclose(result.values, case.reference.exact(nodes), atol=2e-12, rtol=0)
    assert result.relative_residual < 1e-12
    matrix = result.reduced.matrix.toarray()
    np.testing.assert_allclose(matrix, matrix.T, atol=1e-13)
    assert np.linalg.eigvalsh(matrix).min() > 0
    assert np.linalg.eigvalsh(result.assembly.mass.toarray()).min() > 0


def test_mixed_boundary_flux_sign_and_free_neumann_nodes():
    mesh = graded_square_mesh(6, 5, grading=1.3)
    problem = EllipticProblem(
        "linear_mixed", constant(2), constant(0), constant(0),
        BoundaryConditions(constant(1), lambda p: np.isclose(p[:, 0], 0),
                           lambda p, normals: 2*normals[:, 0]),
    )
    result = solve(mesh, problem)
    np.testing.assert_allclose(result.values, 1+mesh.nodes[:, 0], atol=1e-12, rtol=0)
    assert result.assembly.load.sum() == pytest.approx(2.)
    assert len(result.reduced.interior) > np.count_nonzero(~mesh.boundary)
    np.testing.assert_allclose(result.reduced.residual(result.values[result.reduced.interior]),
                               0, atol=1e-12)


@pytest.mark.parametrize("case_name", ["variable_sine", "mixed_polynomial"])
def test_manufactured_convergence_on_graded_mesh(case_name):
    rows = [evaluate(n, n, case_name, grading=1.3)[1] for n in (8, 16, 32)]
    for row in rows:
        assert row["relative_residual"] < 1e-10
        assert row["max_boundary_error"] < 1e-13
    for coarse, fine in zip(rows, rows[1:]):
        for key, lower, upper in (("relative_l2", 1.7, 2.3),
                                  ("relative_h1_seminorm", .8, 1.2)):
            rate = np.log(coarse[key]/fine[key])/np.log(coarse["h_max"]/fine["h_max"])
            assert lower < rate < upper, (case_name, key, rate)


def test_error_quadrature_stabilizes_independently_of_assembly():
    mesh = graded_square_mesh(5, 6, grading=1.3)
    case = get_elliptic_case("variable_sine")
    result = solve(mesh, case.problem)
    low = field_errors(mesh, result.values, case.reference, quadrature_order=5)
    high = field_errors(mesh, result.values, case.reference, quadrature_order=8)
    for key in ("relative_l2", "relative_h1_seminorm"):
        assert low[key] == pytest.approx(high[key], rel=2e-7)


def test_reduced_energy_gradient_and_minimum():
    case = get_elliptic_case("variable_sine")
    result = solve(unit_square_mesh(4), case.problem)
    reduced = result.reduced
    optimum = result.values[reduced.interior]
    direction = np.random.default_rng(3).normal(size=len(optimum))
    trial = optimum + .2*direction
    step = 1e-5
    derivative = (reduced.energy(trial+step*direction)-reduced.energy(trial-step*direction))/(2*step)
    assert derivative == pytest.approx(direction @ reduced.residual(trial), rel=1e-8)
    assert reduced.energy(optimum) < reduced.energy(trial)


@pytest.mark.parametrize("name,value", [("diffusion", 0), ("diffusion", -1),
                                         ("reaction", -1), ("forcing", np.nan),
                                         ("diffusion", np.inf)])
def test_invalid_coefficients_rejected(name, value):
    problem = get_elliptic_case("variable_sine").problem
    with pytest.raises(ValueError):
        solve(unit_square_mesh(3), replace(problem, **{name: constant(value)}))


def test_malformed_or_nonfinite_boundary_data_rejected():
    problem = get_elliptic_case("mixed_polynomial").problem
    mesh = unit_square_mesh(3)
    invalid = [replace(problem.boundary, dirichlet=constant(np.nan)),
               replace(problem.boundary, neumann=lambda p, n: np.full(len(p), np.inf)),
               replace(problem.boundary, neumann=None),
               replace(problem.boundary, dirichlet_selector=lambda p: np.ones(len(p), dtype=int))]
    for boundary in invalid:
        with pytest.raises(ValueError):
            solve(mesh, replace(problem, boundary=boundary))


def test_pure_neumann_is_explicitly_rejected():
    problem = get_elliptic_case("mixed_polynomial").problem
    boundary = replace(problem.boundary, dirichlet_selector=lambda p: np.zeros(len(p), dtype=bool))
    with pytest.raises(ValueError, match="pure Neumann"):
        solve(unit_square_mesh(3), replace(problem, boundary=boundary))


def test_nonuniform_mesh_area_and_invalid_axes():
    mesh = tensor_square_mesh(np.array([0., .1, .4, 1.]), np.array([0., .2, 1.]))
    assert sum(element_geometry(mesh.nodes[t])[0] for t in mesh.triangles) == pytest.approx(1.)
    with pytest.raises(ValueError):
        tensor_square_mesh(np.array([0., .5, .4, 1.]), np.array([0., .5, 1.]))


@pytest.mark.parametrize("vertices", [np.array([[0., 0.], [0., 1.], [1., 0.]]),
                                       np.array([[0., 0.], [1., 0.], [2., 0.]])])
def test_invalid_element_orientation_and_area_rejected(vertices):
    with pytest.raises(ValueError):
        element_geometry(vertices)


def test_mesh_dtypes_and_topological_boundary_validated():
    mesh = unit_square_mesh(3)
    with pytest.raises(ValueError, match="boolean"):
        Mesh2D(mesh.nodes, mesh.triangles, mesh.boundary.astype(int))
    with pytest.raises(ValueError, match="integer"):
        Mesh2D(mesh.nodes, mesh.triangles.astype(float), mesh.boundary)
    wrong_boundary = mesh.boundary.copy()
    wrong_boundary[0] = False
    with pytest.raises(ValueError, match="exterior"):
        solve(Mesh2D(mesh.nodes, mesh.triangles, wrong_boundary),
              get_elliptic_case("variable_sine").problem)


def test_disconnected_component_without_dirichlet_is_rejected():
    left = unit_square_mesh(3)
    nodes = np.vstack((left.nodes, left.nodes + [2, 0]))
    triangles = np.vstack((left.triangles, left.triangles + len(left.nodes)))
    mesh = Mesh2D(nodes, triangles, np.tile(left.boundary, 2))
    problem = EllipticProblem("disconnected", constant(1), constant(0), constant(0),
                              BoundaryConditions(constant(0), lambda p: p[:, 0] < 1.5,
                                                 lambda p, n: np.zeros(len(p))))
    with pytest.raises(ValueError, match="each connected"):
        solve(mesh, problem)


def test_mesh_with_all_prescribed_dofs_needs_no_linear_solve():
    mesh = Mesh2D(np.array([[0., 0.], [1., 0.], [0., 1.]]),
                  np.array([[0, 1, 2]]), np.ones(3, dtype=bool))
    case = get_elliptic_case("affine_diffusion")
    result = solve(mesh, case.problem)
    assert len(result.reduced.interior) == 0
    assert result.absolute_residual == 0
    np.testing.assert_array_equal(result.values, case.reference.exact(mesh.nodes))


def test_underintegrated_mass_matrix_is_rejected():
    with pytest.raises(ValueError, match="at least 2"):
        solve(unit_square_mesh(3), get_elliptic_case("variable_sine").problem,
              quadrature_order=1)
