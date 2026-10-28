"""Local/cloud smoke checks; CLI scripts only orchestrate package calls."""
from pathlib import Path
from time import perf_counter

from physgraph.experiments.elliptic import evaluate as evaluate_elliptic
from physgraph.experiments.pipeline import run_convergence, run_one
from physgraph.io.manifest import write_manifest


def run_cloud_smoke(output: str | Path = "outputs/cloud_smoke") -> Path:
    started = perf_counter()
    dest = Path(output)
    metrics = run_one(24, 24, "sine", dest / "poisson")
    convergence = run_convergence([8, 16, 32], "sine", dest / "convergence")
    _, elliptic = evaluate_elliptic(8, 8, "mixed_polynomial", grading=1.3)
    passed = (metrics["relative_residual"] < 1e-9
              and convergence[-1]["relative_l2"] < convergence[0]["relative_l2"]
              and elliptic["relative_residual"] < 1e-9
              and elliptic["max_boundary_error"] < 1e-12)
    write_manifest(dest / "manifest.json", {"experiment": "cloud_smoke", "phase": "1+2A",
                                             "assembly_quadrature_order_phase2": 4,
                                             "error_quadrature_order": 5},
                   {"single_run": metrics, "mesh_convergence": convergence,
                    "elliptic_mixed": elliptic, "passed": passed}, perf_counter()-started)
    if not passed:
        raise RuntimeError("Numerical smoke checks failed; inspect manifest.json")
    return dest
