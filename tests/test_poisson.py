import numpy as np
import pytest
from dataclasses import replace

from physgraph.experiments.pipeline import evaluate
from physgraph.fem.poisson import assemble, solve
from physgraph.mesh.structured import unit_square_mesh
from physgraph.physics.manufactured import get_case


@pytest.mark.parametrize("case_name", ["sine", "polynomial", "affine"])
def test_all_cases_have_small_interior_residual(case_name):
    field, metrics = evaluate(12, 12, case_name)
    assert metrics["relative_residual"] < 1e-10
    assert np.allclose(field.solution[field.mesh.boundary],
                       field.exact[field.mesh.boundary], atol=1e-12)


def test_affine_nonzero_dirichlet_is_exact():
    field, metrics = evaluate(9, 8, "affine")
    assert metrics["max_nodal_error"] < 1e-11
    assert metrics["relative_h1_seminorm"] < 1e-11
    assert metrics["relative_l2"] < 1e-11


def test_fem_matrix_symmetric_and_positive_on_interior():
    mesh = unit_square_mesh(5, 4)
    k, _ = assemble(mesh, get_case("sine"))
    assert np.max(np.abs((k - k.T).toarray())) < 1e-12
    interior = np.flatnonzero(~mesh.boundary)
    eigs = np.linalg.eigvalsh(k[interior][:, interior].toarray())
    assert eigs.min() > 0


def test_sine_converges_under_mesh_refinement():
    errors = [evaluate(n, n, "sine")[1] for n in (8, 16, 32)]
    l2 = [r["relative_l2"] for r in errors]
    h1 = [r["relative_h1_seminorm"] for r in errors]
    assert l2[0] > l2[1] > l2[2]
    assert h1[0] > h1[1] > h1[2]
    assert l2[0] / l2[1] > 2.5  # expect near 4 for P1 on smooth solutions
    assert h1[0] / h1[1] > 1.6  # expect near 2


def test_load_not_always_zero():
    mesh = unit_square_mesh(4)
    _, load = assemble(mesh, get_case("sine"))
    assert np.linalg.norm(load) > 0


def test_nonfinite_forcing_rejected():
    case = replace(get_case("sine"), forcing=lambda p: np.full(len(p), np.nan))
    with pytest.raises(ValueError, match="finite"):
        solve(unit_square_mesh(4), case)
