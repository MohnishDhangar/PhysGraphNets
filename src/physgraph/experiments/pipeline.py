"""Run validated Poisson experiments without duplicating solver logic."""
import csv
import json
from pathlib import Path
from time import perf_counter

from physgraph.core.types import FieldResult
from physgraph.fem.error import field_errors
from physgraph.fem.poisson import solve
from physgraph.io.results import save_field
from physgraph.io.manifest import write_manifest
from physgraph.mesh.structured import unit_square_mesh
from physgraph.physics.manufactured import get_case
from physgraph.visualization.plotting import plot_comparison, plot_convergence


def evaluate(nx: int, ny: int, case_name: str) -> tuple[FieldResult, dict]:
    mesh = unit_square_mesh(nx, ny)
    case = get_case(case_name)
    fem = solve(mesh, case)
    error = field_errors(mesh, fem.values, case)
    field = FieldResult(mesh=mesh, solution=fem.values, exact=case.exact(mesh.nodes),
                        case=case.name, residual_relative=fem.relative_residual,
                        l2_relative=error["relative_l2"],
                        h1_seminorm_relative=error["relative_h1_seminorm"],
                        linf_nodes=error["max_nodal_error"])
    return field, {"case": case_name, "nx": nx, "ny": ny,
                   "nodes": len(mesh.nodes), "triangles": len(mesh.triangles),
                   "relative_residual": fem.relative_residual, **error}


def run_one(nx: int, ny: int, case_name: str, output: str | Path,
            show_edges: bool = False) -> dict:
    started = perf_counter()
    dest = Path(output)
    dest.mkdir(parents=True, exist_ok=True)
    field, metrics = evaluate(nx, ny, case_name)
    save_field(dest / "field.npz", field)
    plot_comparison(field, dest / "comparison.png", show_edges=show_edges)
    (dest / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    write_manifest(dest / "manifest.json",
                   {"phase": "1", "case": case_name, "nx": nx, "ny": ny,
                    "assembly_quadrature": "symmetric degree 2", "error_quadrature_order": 5,
                    "show_edges": show_edges}, metrics, perf_counter()-started)
    return metrics


def run_convergence(resolutions: list[int], case_name: str,
                    output: str | Path) -> list[dict]:
    started = perf_counter()
    if (len(resolutions) < 2 or any(n < 2 for n in resolutions)
            or any(b <= a for a, b in zip(resolutions, resolutions[1:]))):
        raise ValueError("Supply at least two strictly increasing mesh resolutions >= 2")
    dest = Path(output)
    dest.mkdir(parents=True, exist_ok=True)
    rows = [evaluate(n, n, case_name)[1] for n in resolutions]
    with (dest / "convergence.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_convergence(rows, dest / "convergence.png")
    write_manifest(dest / "convergence_manifest.json",
                   {"phase": "1", "case": case_name, "resolutions": resolutions,
                    "assembly_quadrature": "symmetric degree 2", "error_quadrature_order": 5},
                   rows, perf_counter()-started)
    return rows
