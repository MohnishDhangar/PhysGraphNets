import numpy as np

from physgraph.experiments.pipeline import evaluate, run_one
from physgraph.io.results import load_field


def test_save_load_round_trip(tmp_path):
    original, _ = evaluate(6, 7, "polynomial")
    run_one(6, 7, "polynomial", tmp_path)
    recovered, metadata = load_field(tmp_path / "field.npz")
    assert metadata["schema_version"] == 1
    assert recovered.case == original.case
    assert np.allclose(recovered.solution, original.solution)
    assert np.array_equal(recovered.mesh.triangles, original.mesh.triangles)
    assert (tmp_path / "comparison.png").stat().st_size > 1000
    assert (tmp_path / "metrics.json").stat().st_size > 0
