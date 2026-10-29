import numpy as np
import pytest

from physgraph.fem.elliptic import solve
from physgraph.fem.error import field_errors
from physgraph.mesh.structured import unit_square_mesh
from physgraph.physics.parametric_square import square_case


@pytest.mark.parametrize('parameters', [{}, {'amplitude_1': -.2, 'amplitude_2': .6,
    'diffusion_base': .3, 'diffusion_x': 2., 'diffusion_y': .7, 'reaction': 1.7}])
def test_manufactured_gradient_and_forcing_by_complex_step(parameters):
    case = square_case(**parameters)
    points = np.random.default_rng(42).uniform(.05, .95, (31, 2))
    gradient = np.empty_like(points)
    divergence = np.zeros(len(points))
    for axis in range(2):
        shifted = points.astype(complex)
        shifted[:, axis] += 1e-25j
        gradient[:, axis] = case.reference.exact(shifted).imag/1e-25
        flux = case.problem.diffusion(shifted)*case.reference.gradient(shifted)[:, axis]
        divergence += flux.imag/1e-25
    np.testing.assert_allclose(case.reference.gradient(points), gradient, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(case.problem.forcing(points),
        -divergence+case.problem.reaction(points)*case.reference.exact(points), rtol=3e-14, atol=3e-14)


@pytest.mark.parametrize('mode', ['dirichlet', 'right_neumann'])
def test_parametric_convergence_and_nonzero_dirichlet(mode):
    case = square_case(boundary_mode=mode)
    errors = []
    for n in (8, 16, 32):
        mesh = unit_square_mesh(n)
        solution = solve(mesh, case.problem, 7)
        errors.append(field_errors(mesh, solution.values, case.reference, 7))
        prescribed = solution.reduced.boundary
        np.testing.assert_allclose(solution.values[prescribed], case.reference.exact(mesh.nodes[prescribed]), atol=1e-14)
        assert np.max(np.abs(solution.values[prescribed])) > .5
        assert solution.relative_residual < 1e-11
    for coarse, fine in zip(errors, errors[1:]):
        assert np.log2(coarse['absolute_l2']/fine['absolute_l2']) > 1.8
        assert np.log2(coarse['absolute_h1_seminorm']/fine['absolute_h1_seminorm']) > .9


def test_affine_patch_with_variable_coefficients_and_mixed_flux():
    case = square_case(amplitude_1=0., amplitude_2=0., boundary_mode='right_neumann')
    mesh = unit_square_mesh(5)
    solution = solve(mesh, case.problem, 7)
    np.testing.assert_allclose(solution.values, case.reference.exact(mesh.nodes), atol=2e-14)


@pytest.mark.parametrize('parameters', [{'diffusion_base': 0}, {'reaction': -1},
    {'diffusion_x': -1}, {'amplitude_2': float('nan')}, {'slope_x': True}, {'boundary_mode': 'pure_neumann'}])
def test_invalid_parameters_rejected(parameters):
    with pytest.raises(ValueError):
        square_case(**parameters)
