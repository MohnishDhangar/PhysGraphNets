"""No-slip obstacles, analytical confined-cylinder solution and force checks."""
from dataclasses import replace
from importlib.util import find_spec
import numpy as np
import pytest
from physgraph.physics.obstacle_stokes import ObstacleStokesCase
from physgraph.experiments.obstacle_stokes import evaluate_obstacle, obstacle_mesh
from physgraph.fem.stokes import solve_stokes
from physgraph.mesh.structured import tagged_unit_square_mesh
from physgraph.mesh.quality import mesh_quality


def test_annular_exact_boundary_divergence_momentum_and_drag():
    case = ObstacleStokesCase()
    theta = 2*np.pi*np.arange(256)/256
    normals = np.column_stack((np.cos(theta), np.sin(theta)))
    assert np.max(np.abs(case.fields(case.radius*normals)[0])) < 1e-14
    assert np.max(np.abs(case.fields(case.outer_radius*normals)[0]-[case.speed, 0.])) < 1e-14
    points = normals[::16]*np.linspace(.4, 1.8, 16)[:, None]
    u, grad, _, _ = case.fields(points)
    assert np.max(np.abs(np.trace(grad, axis1=1, axis2=2))) < 1e-14
    h, lap, pg = 1e-5, np.zeros_like(u), np.zeros_like(u)
    for j in range(2):
        delta = h*np.eye(2)[j]
        plus, minus = case.fields(points+delta), case.fields(points-delta)
        assert np.max(np.abs((plus[0]-minus[0])/(2*h)-grad[:, :, j])) < 1e-8
        lap += (plus[1][:, :, j]-minus[1][:, :, j])/(2*h)
        pg[:, j] = (plus[2]-minus[2])/(2*h)
    assert np.max(np.abs(-case.viscosity*lap+pg)) < 1e-8
    _, du, p, _ = case.fields(case.radius*normals)
    stress = case.viscosity*(du+du.transpose(0, 2, 1))-p[:, None, None]*np.eye(2)
    force = 2*np.pi*case.radius*np.einsum('qij,qj->qi', stress, normals).mean(axis=0)
    assert force == pytest.approx([case.analytical_drag, 0.], abs=1e-13)


@pytest.mark.parametrize('name', ['annular_cylinder', 'channel_cylinder', 'airfoil'])
def test_mesh_no_slip_flux_and_positive_drag(name):
    case = ObstacleStokesCase(name)
    result, row = evaluate_obstacle(case, 3)
    assert row['mesh_quality']['holes'] == 1
    assert row['mesh_quality']['euler_characteristic'] == 0
    assert row['mesh_quality']['min_angle_degrees'] > 0
    assert row['max_obstacle_speed'] == 0.
    assert abs(row['boundary_integrals']['net_flux']) < 1e-11
    assert row['reaction_force_on_obstacle'][0] > 0
    assert row['algebraic_force_balance'] < 1e-11
    if name == 'channel_cylinder':
        assert row['boundary_integrals']['by_tag']['left']['flux'] == pytest.approx(-4*case.speed*case.half_height/3, abs=1e-13)
        assert result.diagnostics['pressure_constraint'] == 'traction'
    else:
        assert result.diagnostics['pressure_constraint'] == 'mean_zero'


def test_linearity_of_reference_and_obstacle_load():
    base = ObstacleStokesCase('channel_cylinder')
    first, a = evaluate_obstacle(base, 3)
    second, b = evaluate_obstacle(replace(base, speed=2*base.speed, viscosity=3*base.viscosity), 3)
    assert second.velocity == pytest.approx(2*first.velocity, abs=1e-11)
    assert second.pressure == pytest.approx(6*first.pressure, abs=1e-10)
    assert b['reaction_force_on_obstacle'] == pytest.approx(np.array(a['reaction_force_on_obstacle'])*6, abs=1e-11)


def test_annular_refinement_improves_field_and_force():
    rows = [evaluate_obstacle(ObstacleStokesCase(), n)[1] for n in (2, 4, 8)]
    for key in ('velocity_l2', 'pressure_l2'):
        errors = [row['analytical_errors_on_polygon'][key] for row in rows]
        assert errors[2] < errors[1] < errors[0]
    assert rows[-1]['relative_drag_error'] < rows[0]['relative_drag_error']
    assert rows[-1]['relative_force_disagreement'] < rows[0]['relative_force_disagreement']


def test_mesh_audit_rejects_orientation_and_reports_distortion():
    mesh = obstacle_mesh(ObstacleStokesCase('airfoil'), 3)
    with pytest.raises(ValueError, match='positive oriented'):
        mesh_quality(replace(mesh, triangles=mesh.triangles[:, ::-1]))
    report = mesh_quality(mesh)
    assert report['boundary_signed_areas'][0] < 0 < report['boundary_signed_areas'][1]
    assert 0 < report['min_scaled_quality'] <= 1


def test_airfoil_bias_refines_body_edges_and_preserves_geometry_tags():
    case = ObstacleStokesCase('airfoil')
    uniform = mesh_quality(obstacle_mesh(case, 4))
    biased = mesh_quality(obstacle_mesh(replace(case, airfoil_angular_bias=.2), 4))
    assert uniform['triangles'] == biased['triangles']
    assert biased['min_scaled_quality'] >= 1e-4
    assert biased['holes'] == 1
    for tag in case.obstacle_tags:
        assert biased['boundary_by_tag'][tag]['max_edge'] < uniform['boundary_by_tag'][tag]['max_edge']


def test_boundary_mapping_rejects_conflicting_corners_or_missing_tags():
    mesh = tagged_unit_square_mesh(3)
    zero = lambda x: np.zeros_like(x)
    for data in ({'left': zero}, {'left': lambda x: np.ones_like(x), 'right': zero, 'top': zero, 'bottom': zero}):
        with pytest.raises(ValueError, match='velocity|conflicting'):
            solve_stokes(mesh, zero, data)


@pytest.mark.parametrize('kwargs', [{'name': 'unknown'}, {'speed': 0}, {'viscosity': -1},
                                    {'radius': .8}, {'outer_radius': .1}, {'speed': np.nan},
                                    {'airfoil_angular_bias': 0.}, {'airfoil_angular_bias': 1.1}])
def test_invalid_obstacle_parameters(kwargs):
    with pytest.raises(ValueError):
        ObstacleStokesCase(**kwargs)


@pytest.mark.skipif(find_spec('dolfinx') is None, reason='optional FEniCSx environment')
@pytest.mark.parametrize('name', ['annular_cylinder', 'channel_cylinder', 'airfoil'])
def test_independent_obstacle_fields_fluxes_surface_loads_and_reactions(name):
    _, row = evaluate_obstacle(ObstacleStokesCase(name), 2, external=True)
    assert row['fenicsx']['passed']
