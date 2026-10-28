import json
import subprocess
import sys
from pathlib import Path


def test_script_config(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    config = tmp_path / "experiment.json"
    config.write_text(json.dumps({"case": "affine", "nx": 5, "ny": 5,
                                  "output": str(tmp_path / "results")}))
    result = subprocess.run([sys.executable, "scripts/run_phase1.py",
                             "--config", str(config)], cwd=project_root,
                            capture_output=True, text=True, check=True)
    assert '"case": "affine"' in result.stdout
    assert (tmp_path / "results" / "comparison.png").exists()
