"""Physical patch, pressure-gauge, topology and independent-assembly checks."""
from importlib.util import find_spec
import numpy as np
import pytest
from physgraph.core.types import Mesh2D
from physgraph.mesh.structured import unit_square_mesh
from physgraph.physics.stokes import StokesCase
from physgraph.fem.stokes import solve_stokes
from physgraph.evaluation.stokes import stokes_errors


def solve(case, mesh=None, **kwargs):
    return solve_stokes(unit_square_mesh(4) if mesh is None else mesh,
                        case.forcing, case.velocity, viscosity=case.viscosity, **kwargs)


@pytest.mark.parametrize('name', ['affine', 'channel'])
@pytest.mark.parametrize('viscosity', [.03, 1., 7.])
def test_exact_polynomial_patch_nonzero_boundary_and_pressure_sign(name, viscosity):
    case = StokesCase(name, viscosity)
    result = solve(case)
    assert max(stokes_errors(result, case)[k] for k in
               ('velocity_l2', 'velocity_h1_seminorm', 'pressure_l2', 'divergence_l2')) < 1e-10
    assert result.diagnostics['max_boundary_error'] == 0.
    assert abs(result.diagnostics['pressure_mean']) < 1e-12
    assert abs(result.diagnostics['gauge_multiplier']) < 1e-12


@pytest.mark.parametrize('name', ['vortex', 'translated_vortex'])
def test_manufactured_derivatives_and_forcing(name):
    case = StokesCase(name, .3)
    points = np.random.default_rng(3).uniform(.1, .9, (12, 2))
    u, grad, p, f = case.fields(points)
    lap = np.zeros_like(u)
    pressure_grad = np.zeros_like(u)
    h = 2e-4
    for j in range(2):
        offset = np.eye(2)[j]*h
        plus, minus = case.fields(points+offset), case.fields(points-offset)
        assert np.max(np.abs((plus[0]-minus[0])/(2*h)-grad[:, :, j])) < 1e-7
        lap += (plus[0]-2*u+minus[0])/h**2
        pressure_grad[:, j] = (plus[2]-minus[2])/(2*h)
    assert np.max(np.abs(-case.viscosity*lap+pressure_grad-f)) < 1e-7
    assert np.max(np.abs(np.trace(grad, axis1=1, axis2=2))) < 1e-14


def test_vortex_refinement_and_weak_versus_pointwise_divergence():
    case = StokesCase()
    errors = []
    for n in (4, 8, 16):
        result = solve(case, unit_square_mesh(n))
        assert result.diagnostics['weak_divergence_residual'] < 1e-12
        errors.append(stokes_errors(result, case))
    for key, rate in [('velocity_l2', 2.5), ('velocity_h1_seminorm', 1.8),
                      ('pressure_l2', 1.8), ('divergence_l2', 1.8)]:
        assert np.log2(errors[-2][key]/errors[-1][key]) > rate
    assert errors[-1]['divergence_l2'] > 1e-10  # Taylor-Hood is only weakly solenoidal.


def test_node_permutation_and_ccw_local_rotation():
    case = StokesCase('translated_vortex')
    mesh = unit_square_mesh(4)
    perm = np.random.default_rng(4).permutation(len(mesh.nodes))
    changed = Mesh2D(mesh.nodes[perm], np.argsort(perm)[np.roll(mesh.triangles, 1, axis=1)], mesh.boundary[perm])
    original, result = solve(case, mesh), solve(case, changed)
    assert np.max(np.abs(result.pressure-original.pressure[perm])) < 1e-11
    for key, value in stokes_errors(original, case).items():
        assert stokes_errors(result, case)[key] == pytest.approx(value, rel=1e-8, abs=1e-12)


def test_reject_clockwise_and_wrong_boundary():
    mesh = unit_square_mesh(4)
    with pytest.raises(ValueError, match='clockwise'):
        solve(StokesCase(), Mesh2D(mesh.nodes, mesh.triangles[:, ::-1], mesh.boundary))
    with pytest.raises(ValueError, match='boundary mask'):
        solve(StokesCase(), Mesh2D(mesh.nodes, mesh.triangles, np.zeros(len(mesh.nodes), bool)))


def test_reject_flux_incompatibility_instead_of_hiding_it_in_gauge():
    with pytest.raises(ValueError, match='boundary flux'):
        solve_stokes(unit_square_mesh(4), lambda x: np.zeros_like(x), lambda x: x)


@pytest.mark.parametrize('value', [0., -1., np.nan, np.inf])
def test_invalid_viscosity(value):
    with pytest.raises(ValueError, match='viscosity'):
        StokesCase(viscosity=value)


def test_reject_nonfinite_load_and_bad_quadrature():
    with pytest.raises(ValueError, match='forcing'):
        solve_stokes(unit_square_mesh(4), lambda x: np.full_like(x, np.nan), lambda x: x*0)
    with pytest.raises(ValueError, match='quadrature'):
        solve(StokesCase(), quadrature_order=2)


@pytest.mark.skipif(find_spec('dolfinx') is None, reason='optional FEniCSx environment')
@pytest.mark.parametrize('name', ['channel', 'translated_vortex'])
def test_independent_ufl_assembly_and_pinned_pressure_gauge(name):
    from physgraph.verification.stokes import compare_stokes
    case = StokesCase(name, .3)
    assert compare_stokes(solve(case), case)['passed']
