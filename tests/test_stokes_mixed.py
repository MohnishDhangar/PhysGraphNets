"""Physical traction, absolute pressure, flux and load verification."""
from importlib.util import find_spec
import numpy as np
import pytest
from physgraph.core.types import Mesh2D
from physgraph.mesh.structured import tagged_unit_square_mesh, unit_square_mesh
from physgraph.mesh.cylinder import cylinder_mesh
from physgraph.physics.stokes import StokesCase
from physgraph.fem.stokes import solve_stokes
from physgraph.evaluation.stokes import stokes_errors, stokes_boundary_integrals
from physgraph.fem.stokes_boundary import stokes_facets


def mixed(case, tag='right', n=4, form='symmetric_gradient', mesh=None):
    mesh = tagged_unit_square_mesh(n) if mesh is None else mesh
    selected = set(mesh.boundary_edge_tags)-{tag}
    return solve_stokes(mesh, case.forcing, case.velocity, viscosity=case.viscosity,
        viscous_form=form, dirichlet_tags=selected,
        tractions={tag: lambda x, normal: case.traction(x, normal, form)})


@pytest.mark.parametrize('tag', ['left', 'right', 'top', 'bottom'])
@pytest.mark.parametrize('name', ['affine', 'channel'])
def test_patch_mixed_normals_and_pressure_level(tag, name):
    case = StokesCase(name, .3, 2.5)
    result = mixed(case, tag)
    errors = stokes_errors(result, case)
    assert max(errors[k] for k in ('velocity_l2', 'velocity_h1_seminorm', 'pressure_l2')) < 1e-10
    assert result.diagnostics['pressure_constraint'] == 'traction'
    assert result.diagnostics['gauge_multiplier'] is None
    assert result.diagnostics['pressure_mean'] == pytest.approx(2.5, abs=1e-11)
    boundary = stokes_boundary_integrals(result, case)
    assert abs(boundary['net_flux']) < 1e-12
    assert boundary['traction_l2_error'] < 1e-10
    assert boundary['stress_force_balance'] < 1e-10
    assert result.diagnostics['algebraic_force_balance'] < 1e-11
    if name == 'channel':
        assert boundary['by_tag']['left']['flux'] == pytest.approx(-1/6, abs=1e-12)
        assert boundary['by_tag']['right']['flux'] == pytest.approx(1/6, abs=1e-12)
        # Physical normal stress and tangential shear at the right face.
        assert boundary['by_tag']['right']['force_on_fluid'] == pytest.approx([.3-2.5, 0.], abs=1e-11)


def test_pressure_offset_changes_load_but_not_velocity():
    base, shifted = StokesCase('translated_vortex'), StokesCase('translated_vortex', pressure_offset=4.)
    first, second = mixed(base), mixed(shifted)
    assert np.max(np.abs(first.velocity-second.velocity)) < 1e-11
    assert np.max(np.abs(second.pressure-first.pressure-4)) < 1e-10


def test_zero_physical_traction_patch():
    # Extension u=(1+x,-.5-y) has sigma=diag(0,-4nu) when p=2nu.
    # Thus right-face traction is zero although pressure is nonzero.
    nu = .7
    velocity = StokesCase('affine').velocity
    result = solve_stokes(tagged_unit_square_mesh(4), lambda x: np.zeros_like(x), velocity,
        viscosity=nu, viscous_form='symmetric_gradient', dirichlet_tags={'left', 'top', 'bottom'},
        tractions={'right': lambda x, n: np.zeros_like(x)})
    assert np.max(np.abs(result.velocity-velocity(result.space.points))) < 1e-11
    assert np.max(np.abs(result.pressure-2*nu)) < 1e-10


def test_physical_channel_traction_includes_tangential_shear():
    case = StokesCase('channel')
    xy = np.array([[1., .2], [1., .8]])
    normals = np.tile([1., 0.], (2, 1))
    physical = case.traction(xy, normals)
    conormal = case.traction(xy, normals, 'laplacian')
    assert physical[:, 1] == pytest.approx([.6, -.6])
    assert np.all(conormal[:, 1] == 0)
    assert np.allclose(physical[:, 0], conormal[:, 0])


def test_stress_versus_laplacian_form_is_explicit():
    case = StokesCase('channel', .4, 1.2)
    result = mixed(case, form='laplacian')
    assert stokes_errors(result, case)['pressure_l2'] < 1e-11
    assert stokes_boundary_integrals(result, case)['traction_l2_error'] < 1e-10


def test_symmetric_gradient_full_boundary_uses_mean_gauge():
    case = StokesCase('affine', .3)
    result = solve_stokes(unit_square_mesh(4), case.forcing, case.velocity,
                          viscosity=.3, viscous_form='symmetric_gradient')
    assert result.diagnostics['pressure_constraint'] == 'mean_zero'
    assert stokes_errors(result, case)['pressure_l2'] < 1e-11


@pytest.mark.parametrize('selected,tractions', [
    ({'left', 'top', 'bottom'}, {}),
    ({'left', 'top', 'bottom'}, {'right': None}),
    ({'left', 'top', 'bottom'}, {'left': lambda x, n: x, 'right': lambda x, n: x}),
    (set(), {}), ({'typo'}, {}), ('left', {}),
])
def test_reject_incomplete_overlapping_or_unknown_partition(selected, tractions):
    case = StokesCase()
    with pytest.raises(ValueError):
        solve_stokes(tagged_unit_square_mesh(4), case.forcing, case.velocity,
                      dirichlet_tags=selected, tractions=tractions)


def test_reject_untagged_mixed_boundary_and_nonfinite_traction():
    case = StokesCase()
    for mesh, traction in [(unit_square_mesh(4), lambda x, n: x),
                           (tagged_unit_square_mesh(4), lambda x, n: np.full_like(x, np.nan))]:
        with pytest.raises(ValueError):
            solve_stokes(mesh, case.forcing, case.velocity, dirichlet_tags={'left', 'top', 'bottom'},
                          tractions={'right': traction})


def test_boundary_normal_orientation_on_hole_and_reversed_tagged_edges():
    mesh = cylinder_mesh(2, 4)
    reverse = Mesh2D(mesh.nodes, mesh.triangles, mesh.boundary, mesh.boundary_edges[:, ::-1], mesh.boundary_edge_tags)
    facets = stokes_facets(reverse)
    for facet in facets:
        if facet.tag == 'cylinder':
            midpoint = mesh.nodes[list(facet.vertices)].mean(axis=0)
            assert midpoint @ facet.normal < 0  # outward from fluid points into hole
    case = StokesCase('affine', .3, 2.5)
    # Manufactured topology check only: this velocity is NOT no-slip on the obstacle.
    result = mixed(case, mesh=reverse)
    assert stokes_errors(result, case)['velocity_l2'] < 1e-11
    assert stokes_boundary_integrals(result, case)['max_tag_force_error'] < 1e-10


def test_edge_reversal_local_rotation_and_permutation():
    mesh = tagged_unit_square_mesh(4)
    perm = np.random.default_rng(4).permutation(len(mesh.nodes))
    inverse = np.argsort(perm)
    changed = Mesh2D(mesh.nodes[perm], inverse[np.roll(mesh.triangles, 1, axis=1)], mesh.boundary[perm],
                     inverse[mesh.boundary_edges[::-1, ::-1]], mesh.boundary_edge_tags[::-1])
    case = StokesCase('translated_vortex', .3, 2.5)
    original, result = mixed(case, mesh=mesh), mixed(case, mesh=changed)
    assert np.max(np.abs(result.pressure-original.pressure[perm])) < 1e-11
    a, b = stokes_boundary_integrals(original), stokes_boundary_integrals(result)
    for tag in a['by_tag']:
        assert a['by_tag'][tag]['flux'] == pytest.approx(b['by_tag'][tag]['flux'], abs=1e-12)
        assert a['by_tag'][tag]['force_on_fluid'] == pytest.approx(b['by_tag'][tag]['force_on_fluid'], abs=1e-11)


def test_vortex_convergence():
    case = StokesCase('translated_vortex', .3, 2.5)
    errors = []
    for n in (4, 8, 16):
        result = mixed(case, n=n)
        errors.append({**stokes_errors(result, case), **stokes_boundary_integrals(result, case)})
        assert abs(errors[-1]['net_flux']) < 1e-11
    for key, minimum in [('velocity_l2', 2.5), ('velocity_h1_seminorm', 1.8),
                         ('pressure_l2', 1.8), ('traction_l2_error', 1.2)]:
        assert np.log2(errors[-2][key]/errors[-1][key]) > minimum


@pytest.mark.skipif(find_spec('dolfinx') is None, reason='optional FEniCSx environment')
@pytest.mark.parametrize('tag', ['left', 'right'])
def test_ufl_mixed_fields_fluxes_loads(tag):
    from physgraph.verification.stokes import compare_stokes
    case = StokesCase('translated_vortex', .3, 2.5)
    assert compare_stokes(mixed(case, tag), case)['passed']
