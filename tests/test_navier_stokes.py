"""Continuous forcing, implicit advection, time order and boundary contracts."""
from dataclasses import replace
from importlib.util import find_spec
import numpy as np
import pytest
from physgraph.core.types import Mesh2D
from physgraph.mesh.structured import tagged_unit_square_mesh
from physgraph.physics.navier_stokes import NavierStokesCase
from physgraph.physics.stokes import StokesCase
from physgraph.fem.stokes import solve_stokes, taylor_hood_space
from physgraph.fem.flow_operators import FlowOperators
from physgraph.fem.navier_stokes import solve_navier_stokes
from physgraph.evaluation.stokes import stokes_errors, stokes_boundary_integrals


def run(case, *, n=4, dt=.05, steps=3, scheme='bdf2', traction_tag=None, mesh=None, **kwargs):
    mesh = tagged_unit_square_mesh(n) if mesh is None else mesh
    return solve_navier_stokes(mesh, case.forcing, case.velocity, lambda x: case.velocity(x, 0.),
        viscosity=case.viscosity, dt=dt, steps=steps, scheme=scheme,
        dirichlet_tags=None if traction_tag is None else set(mesh.boundary_edge_tags)-{traction_tag},
        tractions=None if traction_tag is None else {traction_tag: case.traction}, **kwargs)


@pytest.mark.parametrize('name', ['accelerating_shear', 'vortex', 'translated_vortex'])
@pytest.mark.parametrize('profile', ['linear', 'exponential'])
def test_continuous_momentum_time_derivative_and_incompressibility(name, profile):
    case = NavierStokesCase(name, time_profile=profile, pressure_offset=2.5)
    points = np.random.default_rng(8).uniform(.1, .9, (15, 2))
    h, t = 1e-5, .2
    u, grad, _, f = case.fields(points, t)
    ut = (case.velocity(points, t+h)-case.velocity(points, t-h))/(2*h)
    lap, gp = np.zeros_like(u), np.zeros_like(u)
    for j in range(2):
        offset = h*np.eye(2)[j]
        plus, minus = case.fields(points+offset, t), case.fields(points-offset, t)
        assert np.max(np.abs((plus[0]-minus[0])/(2*h)-grad[:, :, j])) < 1e-7
        lap += (plus[1][:, :, j]-minus[1][:, :, j])/(2*h)
        gp[:, j] = (plus[2]-minus[2])/(2*h)
    assert np.max(np.abs(np.trace(grad, axis1=1, axis2=2))) < 1e-14
    advection = np.einsum('qij,qj->qi', grad, u)
    assert np.linalg.norm(advection) > .01
    assert np.max(np.abs(ut+advection-case.viscosity*lap+gp-f)) < 2e-7


def test_mass_stress_energy_and_convection_jacobian():
    space = taylor_hood_space(tagged_unit_square_mesh(3))
    operators = FlowOperators(space, .3)
    one = np.ones(2*len(space.points))
    assert one @ operators.mass @ one == pytest.approx(2., abs=1e-13)
    extension = np.r_[space.points[:, 0], -space.points[:, 1]]
    assert extension @ operators.viscous @ extension == pytest.approx(1.2, abs=1e-12)
    rng = np.random.default_rng(9)
    u, direction = rng.normal(size=(2, len(one)))
    _, jacobian = operators.convection(u, jacobian=True)
    h = 1e-6
    difference = (operators.convection(u+h*direction)-operators.convection(u-h*direction))/(2*h)
    assert np.max(np.abs(difference-jacobian @ direction)) < 1e-9


@pytest.mark.parametrize('scheme', ['be', 'bdf2'])
@pytest.mark.parametrize('tag', [None, 'left', 'right', 'top', 'bottom'])
def test_exact_time_dependent_patch_all_normals_and_pressure_level(scheme, tag):
    case = NavierStokesCase('accelerating_shear', time_profile='linear', pressure_offset=0. if tag is None else 2.5)
    result = run(case, scheme=scheme, traction_tag=tag)
    for step in range(1, len(result.times)):
        frame = result.frame(step)
        errors = stokes_errors(frame, case.at(result.times[step]))
        assert max(errors[k] for k in ('velocity_l2', 'velocity_h1_seminorm', 'pressure_l2')) < 1e-10
        assert frame.diagnostics['pressure_mean'] == pytest.approx(case.pressure_offset, abs=1e-10)
        assert frame.diagnostics['max_boundary_error'] == 0.
        assert frame.diagnostics['weak_divergence_residual'] < 1e-12
        assert frame.diagnostics['algebraic_force_balance'] < 1e-10
        integrals = stokes_boundary_integrals(frame, case.at(result.times[step]))
        assert integrals['balance_kind'] == 'unsteady_momentum'
        assert integrals['stress_force_balance'] < 1e-10
    assert result.initial_diagnostics['weak_divergence_residual'] < 1e-12
    assert len(result.pressure) == len(result.times)-1
    with pytest.raises(ValueError, match='pressure'):
        result.frame(0)


def test_stationary_channel_matches_verified_stokes_reference():
    mesh = tagged_unit_square_mesh(4)
    case = StokesCase('channel', .3)
    steady = solve_stokes(mesh, case.forcing, case.velocity, viscosity=.3, viscous_form='symmetric_gradient')
    transient = solve_navier_stokes(mesh, lambda x,t: case.forcing(x), lambda x,t: case.velocity(x),
        case.velocity, dt=.1, steps=3, viscosity=.3)
    assert transient.velocity[-1] == pytest.approx(steady.velocity, abs=1e-11)
    assert transient.pressure[-1] == pytest.approx(steady.pressure, abs=1e-10)


@pytest.mark.parametrize('scheme,minimum', [('be', .85), ('bdf2', 1.75)])
def test_time_refinement_without_spatial_approximation_error(scheme, minimum):
    case = NavierStokesCase('accelerating_shear')
    errors = []
    for steps in (8, 16, 32):
        result = run(case, dt=.4/steps, steps=steps, scheme=scheme)
        errors.append(stokes_errors(result.frame(steps), case.at(.4)))
    for key in ('velocity_l2', 'pressure_l2'):
        assert np.log2(errors[-2][key]/errors[-1][key]) > minimum


def test_projection_removes_interpolation_divergence_and_pressure_shift_does_not_change_velocity():
    case = NavierStokesCase('translated_vortex', time_profile='linear')
    first = run(case, traction_tag='right')
    second = run(replace(case, pressure_offset=3.), traction_tag='right')
    assert np.max(np.abs(first.velocity-second.velocity)) < 1e-10
    assert np.max(np.abs(second.pressure-first.pressure-3.)) < 1e-9
    operators = FlowOperators(first.space, case.viscosity)
    interpolated = case.velocity(first.space.points, 0).T.ravel()
    assert np.linalg.norm(operators.divergence @ interpolated) > 1e-6
    assert first.initial_diagnostics['weak_divergence_residual'] < 1e-12


def test_node_permutation_triangle_rotation_and_reversed_tags():
    mesh = tagged_unit_square_mesh(3)
    perm = np.random.default_rng(10).permutation(len(mesh.nodes))
    inverse = np.argsort(perm)
    changed = Mesh2D(mesh.nodes[perm], inverse[np.roll(mesh.triangles, 1, axis=1)], mesh.boundary[perm],
        inverse[mesh.boundary_edges[::-1, ::-1]], mesh.boundary_edge_tags[::-1])
    case = NavierStokesCase('translated_vortex', time_profile='linear', pressure_offset=2.5)
    original, other = run(case, mesh=mesh, traction_tag='right'), run(case, mesh=changed, traction_tag='right')
    assert np.max(np.abs(other.pressure-original.pressure[:, perm])) < 1e-10
    assert np.max(np.abs(other.velocity[:, :len(perm)]-original.velocity[:, perm])) < 1e-11


def test_unforced_closed_wall_energy_decay_and_balance():
    mesh = tagged_unit_square_mesh(5)
    case = NavierStokesCase('vortex')
    zero = lambda x,t: np.zeros_like(x)
    result = solve_navier_stokes(mesh, zero, zero, lambda x: case.velocity(x,0), dt=.02, steps=8, scheme='be')
    ops = FlowOperators(result.space, result.viscosity)
    energy = [.5*result.velocity[0].T.ravel() @ ops.mass @ result.velocity[0].T.ravel()]
    for step, row in enumerate(result.diagnostics, 1):
        u, before = result.velocity[step].T.ravel(), result.velocity[step-1].T.ravel()
        energy.append(row['kinetic_energy'])
        balance = (energy[-1]-energy[-2])/.02 + .5*(u-before) @ ops.mass @ (u-before)/.02
        balance += row['viscous_dissipation']+row['convective_power']
        assert abs(balance) < 1e-10
    assert np.all(np.diff(energy) < 0)


@pytest.mark.parametrize('scheme', ['be', 'bdf2'])
def test_energy_evaluation_and_trajectory_archive(scheme, tmp_path):
    from physgraph.evaluation.navier_stokes import closed_wall_energy
    from physgraph.experiments.navier_stokes import save_trajectory
    case = NavierStokesCase('vortex')
    zero = lambda x,t: np.zeros_like(x)
    result = solve_navier_stokes(tagged_unit_square_mesh(3), zero, zero,
        lambda x:case.velocity(x,0), dt=.025,steps=4,scheme=scheme)
    energy = closed_wall_energy(result)
    assert energy['max_energy_increase'] < 0
    assert energy['max_absolute_balance'] < 1e-10
    path = tmp_path/'trajectory.npz'
    save_trajectory(path,result,{'case':'unforced'})
    with np.load(path,allow_pickle=False) as data:
        assert np.array_equal(data['pressure_times'],data['times'][1:])
        assert data['velocity'].shape[0]==5
        assert data['pressure'].shape[0]==4
        assert np.array_equal(data['velocity'],result.velocity)
        assert 'projection multiplier' in str(data['metadata'])


def test_energy_diagnostic_rejects_nonzero_load_even_when_net_force_is_zero():
    from physgraph.evaluation.navier_stokes import closed_wall_energy
    zero = lambda x,t: np.zeros_like(x)
    result = solve_navier_stokes(tagged_unit_square_mesh(3), lambda x,t:x-.5, zero,
                                lambda x:x*0,dt=.1,steps=1)
    assert np.linalg.norm(result.diagnostics[0]['body_force']) < 1e-12
    with pytest.raises(ValueError,match='zero applied loads'):
        closed_wall_energy(result)


def test_time_dependent_boundary_mapping_and_conflicting_corners():
    case = NavierStokesCase('accelerating_shear',time_profile='linear')
    mesh = tagged_unit_square_mesh(3)
    mapping = {tag:case.velocity for tag in ('left','right','top','bottom')}
    result = solve_navier_stokes(mesh,case.forcing,mapping,lambda x:case.velocity(x,0),dt=.1,steps=2)
    assert stokes_errors(result.frame(2),case.at(.2))['velocity_l2'] < 1e-11
    mapping['left'] = lambda x,t:case.velocity(x,t)+t
    with pytest.raises(ValueError,match='conflicting'):
        solve_navier_stokes(mesh,case.forcing,mapping,lambda x:case.velocity(x,0),dt=.1,steps=1)


@pytest.mark.parametrize('kwargs', [{'dt':0.}, {'dt':np.nan}, {'steps':0}, {'steps':True},
    {'scheme':'invalid'}, {'viscosity':0.}, {'quadrature_order':3}, {'newton_tolerance':0.},
    {'initial_time':-1.}, {'max_iterations':0}])
def test_invalid_configuration(kwargs):
    settings = dict(dt=.1,steps=2)
    settings.update(kwargs)
    case = NavierStokesCase()
    with pytest.raises(ValueError):
        solve_navier_stokes(tagged_unit_square_mesh(3),case.forcing,case.velocity,lambda x:case.velocity(x,0),**settings)


def test_incompatible_initial_boundary_flux_and_failed_newton():
    case = NavierStokesCase('accelerating_shear')
    mesh = tagged_unit_square_mesh(3)
    with pytest.raises(ValueError, match='initial velocity'):
        solve_navier_stokes(mesh,case.forcing,case.velocity,lambda x:x*0,dt=.1,steps=1)
    # Initially compatible data become incompatible at t>0; check every step.
    with pytest.raises(ValueError, match='boundary flux'):
        solve_navier_stokes(mesh,lambda x,t:x*0,lambda x,t:t*x,lambda x:x*0,dt=.1,steps=1)
    with pytest.raises(RuntimeError, match='Newton failed'):
        run(case,max_iterations=1)
    with pytest.raises(ValueError, match='volume data'):
        solve_navier_stokes(mesh,lambda x,t:x*np.nan,case.velocity,lambda x:case.velocity(x,0),dt=.1,steps=1)


@pytest.mark.skipif(find_spec('dolfinx') is None, reason='optional FEniCSx environment')
@pytest.mark.parametrize('tag', [None, 'right'])
def test_independent_ufl_snes_projection_fields_fluxes_loads_and_reactions(tag):
    from physgraph.verification.navier_stokes import compare_navier_stokes
    case = NavierStokesCase('translated_vortex',time_profile='linear',pressure_offset=0. if tag is None else 2.5)
    trajectory = run(case,n=3,steps=2,traction_tag=tag)
    assert compare_navier_stokes(trajectory,case)['passed']
