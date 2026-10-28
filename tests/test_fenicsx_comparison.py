import json
from importlib.util import find_spec
import numpy as np
import pytest

from physgraph.verification.cases import comparison_case, CASE_IDS
from physgraph.verification.fenicsx import coordinate_permutation, compare
from physgraph.experiments.fenicsx import run_comparisons

HAS_FENICSX = all(find_spec(name) is not None for name in ('dolfinx', 'ufl', 'basix', 'mpi4py', 'petsc4py'))


def test_coordinate_reordering_and_non_bijections():
    points = np.array([[0., 0.], [1., 0.], [0., 1.]])
    np.testing.assert_array_equal(coordinate_permutation(points, points[[2, 0, 1]]), [2, 0, 1])
    for invalid in (points[[0, 0, 2]], points+.1, points[:2]):
        with pytest.raises(ValueError):
            coordinate_permutation(points, invalid)


@pytest.mark.parametrize('name', CASE_IDS)
def test_comparison_fixtures_are_finite_and_oriented(name):
    case = comparison_case(name, 2)
    vertices = case.mesh.nodes[case.mesh.triangles]
    a, b = vertices[:, 1]-vertices[:, 0], vertices[:, 2]-vertices[:, 0]
    assert np.all(a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0] > 0)
    assert np.isfinite(case.manufactured.reference.exact(case.mesh.nodes)).all()


@pytest.mark.skipif(HAS_FENICSX, reason='this test exercises the missing-dependency path')
def test_missing_fenicsx_is_recorded_as_blocked_not_passed(tmp_path):
    result = run_comparisons(cases=['affine_diffusion'], resolutions=[2], output=tmp_path/'blocked')
    assert result['metrics']['status'] == 'blocked_missing_dependency'
    assert result['metrics']['comparisons'] == []
    assert 'Missing' in result['metrics']['reason']
    saved = json.loads((tmp_path/'blocked'/'manifest.json').read_text())
    assert saved['python'] and 'code_revision' in saved


@pytest.mark.skipif(not HAS_FENICSX, reason='independent FEniCSx/PETSc environment is not installed')
@pytest.mark.parametrize('name', CASE_IDS)
def test_actual_fenicsx_assembly_and_solution_agree(name, tmp_path):
    metrics, arrays = compare(comparison_case(name, 3), cache_dir=tmp_path/'jit')
    assert metrics['passed'], metrics
    assert metrics['max_nodal_difference'] < 1e-8
    np.testing.assert_allclose(metrics['fenicsx_independent_absolute_l2'],
                               metrics['fenicsx_nodal_field_errors']['absolute_l2'], atol=1e-8, rtol=1e-6)
    np.testing.assert_allclose(metrics['fenicsx_independent_absolute_h1'],
                               metrics['fenicsx_nodal_field_errors']['absolute_h1_seminorm'], atol=1e-8, rtol=1e-6)
    assert arrays['fenicsx_fem'].shape == arrays['physgraph_fem'].shape
