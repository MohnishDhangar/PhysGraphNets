import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from physgraph.io.results import load_field
from physgraph.experiments.elliptic import run_experiment


def test_phase2_cli_config_and_provenance(tmp_path):
    config = tmp_path / "config.json"
    dest = tmp_path / "result"
    config.write_text(json.dumps({"case": "mixed_polynomial", "nx": 4,
                                  "grading": 1.3, "output": str(dest)}))
    project = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "scripts/run_phase2.py", "--config", str(config),
                             "--nx", "5", "--convergence", "4", "8"], cwd=project,
                            text=True, capture_output=True, check=True)
    assert "Saved experiment:" in result.stdout
    manifest = json.loads((dest / "manifest.json").read_text())
    assert manifest["config"]["nx"] == manifest["config"]["ny"] == 5
    assert manifest["python"]
    assert manifest["dependencies"]["scipy"]
    assert "code_revision" in manifest and "git_dirty" in manifest
    assert len(manifest["source_sha256"]) == 64
    assert len(manifest["metrics"]["convergence"]) == 2
    assert manifest["metrics"]["convergence"][1]["l2_rate"] > 1.5
    field, metadata = load_field(dest / "field.npz")
    assert metadata["equation"] == "-div(k grad(u)) + c*u = f"
    assert field.boundary_type == "mixed Dirichlet/Neumann"
    assert np.isfinite(field.solution).all()
    for filename in ("comparison.png", "convergence.csv", "convergence.png"):
        assert (dest / filename).stat().st_size > 0
    with pytest.raises(ValueError, match="preserve"):
        run_experiment(output=dest)


def test_nonfinite_archive_is_rejected(tmp_path):
    dest = tmp_path / "result"
    run_experiment(nx=3, case="affine_diffusion", output=dest)
    with np.load(dest / "field.npz", allow_pickle=False) as archive:
        payload = {name: archive[name].copy() for name in archive.files}
    payload["solution"][0] = np.nan
    np.savez_compressed(dest / "invalid.npz", **payload)
    with pytest.raises(ValueError, match="finite"):
        load_field(dest / "invalid.npz")
