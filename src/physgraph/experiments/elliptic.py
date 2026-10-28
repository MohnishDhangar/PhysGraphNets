"""Phase 2A experiments: shared CPU code for scripts and cloud notebooks."""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np

from physgraph.core.types import FieldResult
from physgraph.fem.elliptic import solve
from physgraph.fem.error import field_errors
from physgraph.io.manifest import write_manifest
from physgraph.io.results import save_field
from physgraph.mesh.nonuniform import graded_square_mesh
from physgraph.physics.diffusion import CASE_NAMES, get_elliptic_case
from physgraph.visualization.plotting import plot_comparison, plot_convergence


def evaluate(nx: int, ny: int, case_name: str, grading: float = 1.,
             quadrature_order: int = 4, error_quadrature_order: int = 5) -> tuple[FieldResult, dict]:
    mesh = graded_square_mesh(nx, ny, grading)
    case = get_elliptic_case(case_name)
    fem = solve(mesh, case.problem, quadrature_order)
    errors = field_errors(mesh, fem.values, case.reference, error_quadrature_order)
    exact = case.reference.exact(mesh.nodes)
    field = FieldResult(mesh, fem.values, exact, case_name, fem.relative_residual,
                        errors["relative_l2"], errors["relative_h1_seminorm"],
                        errors["max_nodal_error"], "-div(k grad(u)) + c*u = f",
                        "mixed Dirichlet/Neumann" if len(fem.assembly.boundary.neumann_edges)
                        else "Dirichlet")
    vertices = mesh.nodes[mesh.triangles]
    h_max = float(np.max(np.linalg.norm(vertices - np.roll(vertices, 1, axis=1), axis=2)))
    metrics = {"case": case_name, "nx": nx, "ny": ny, "grading": grading,
               "nodes": len(mesh.nodes), "triangles": len(mesh.triangles), "h_max": h_max,
               "dirichlet_nodes": len(fem.reduced.boundary),
               "neumann_edges": len(fem.assembly.boundary.neumann_edges),
               "max_boundary_error": float(np.max(np.abs(fem.values[fem.reduced.boundary]
                                                         - exact[fem.reduced.boundary]))),
               "absolute_residual": fem.absolute_residual,
               "relative_residual": fem.relative_residual, **errors}
    return field, metrics


def run_experiment(*, case: str = "variable_sine", nx: int = 16,
                   ny: int | None = None, grading: float = 1.,
                   output: str | Path | None = None,
                   convergence: list[int] | None = None,
                   quadrature_order: int = 4, error_quadrature_order: int = 5) -> dict:
    """Write a single solution and optionally a separate square-mesh refinement study.

    Explicit output directories must be empty. Defaults create a unique run
    directory so experiments never silently replace previous evidence.
    """
    started = perf_counter()
    ny = nx if ny is None else ny
    if case not in CASE_NAMES:
        raise ValueError(f"case must be one of {CASE_NAMES}")
    if convergence is not None:
        if (len(convergence) < 2
                or any(isinstance(n, bool) or not isinstance(n, int) or n < 2 for n in convergence)
                or any(b <= a for a, b in zip(convergence, convergence[1:]))):
            raise ValueError("convergence sizes must be strictly increasing integers >= 2")
    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = Path("outputs/phase2") / f"{case}_{stamp}_{uuid4().hex[:8]}"
    dest = Path(output)
    if dest.exists() and (not dest.is_dir() or any(dest.iterdir())):
        raise ValueError(f"output directory must be empty to preserve previous runs: {dest}")
    field, metrics = evaluate(nx, ny, case, grading, quadrature_order, error_quadrature_order)
    dest.mkdir(parents=True, exist_ok=True)
    save_field(dest / "field.npz", field)
    plot_comparison(field, dest / "comparison.png")
    (dest / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    rows = []
    for n in convergence or []:
        row = evaluate(n, n, case, grading, quadrature_order, error_quadrature_order)[1]
        for key, rate in (("relative_l2", "l2_rate"), ("relative_h1_seminorm", "h1_rate")):
            row[rate] = None
            if rows and row[key] > 1e-12 and rows[-1][key] > 1e-12:
                row[rate] = float(np.log(rows[-1][key]/row[key])
                                  / np.log(rows[-1]["h_max"]/row["h_max"]))
        rows.append(row)
    if rows:
        with (dest / "convergence.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        plot_convergence(rows, dest / "convergence.png")
    config = {"phase": "2A", "case": case, "nx": nx, "ny": ny, "grading": grading,
              "convergence": convergence, "quadrature_order": quadrature_order,
              "error_quadrature_order": error_quadrature_order,
              "output": str(dest.resolve()), "random_seed": None}
    return write_manifest(dest / "manifest.json", config,
                          {"single_run": metrics, "convergence": rows}, perf_counter()-started)
