"""Rectangle/cylinder verification with independent manufactured fields."""
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
from physgraph.mesh.cylinder import cylinder_mesh
from physgraph.physics.cylinder import cylinder_case
from physgraph.visualization.plotting import plot_comparison, plot_convergence


def evaluate(nr: int, nside: int, *, radius: float = .3,
             bounds=(-1., 1., -.75, .75), center=(0., 0.), grading: float = 1.,
             case: str = "quadratic", diffusivity: float = 1.,
             boundary_mode: str = "dirichlet", quadrature_order: int = 4,
             error_quadrature_order: int = 5) -> tuple[FieldResult, dict]:
    mesh = cylinder_mesh(nr, nside, radius, bounds, center, grading)
    case = cylinder_case(case, boundary_mode, diffusivity)
    solution = solve(mesh, case.problem, quadrature_order)
    errors = field_errors(mesh, solution.values, case.reference, error_quadrature_order)
    exact = case.reference.exact(mesh.nodes)
    field = FieldResult(mesh, solution.values, exact, case.problem.name,
                        solution.relative_residual, errors["relative_l2"],
                        errors["relative_h1_seminorm"], errors["max_nodal_error"],
                        "-div(k grad(u)) + u = f",
                        "Dirichlet" if boundary_mode == "dirichlet" else "mixed Dirichlet/Neumann")
    vertices = mesh.nodes[mesh.triangles]
    first, second = vertices[:, 1]-vertices[:, 0], vertices[:, 2]-vertices[:, 0]
    areas = .5*(first[:, 0]*second[:, 1]-first[:, 1]*second[:, 0])
    area = float(np.sum(areas))
    rectangle_area = (bounds[1]-bounds[0])*(bounds[3]-bounds[2])
    circular_area = float(rectangle_area-np.pi*radius**2)
    inner = mesh.nodes[:4*nside]-np.asarray(center)
    nxt = np.roll(inner, -1, axis=0)
    hole_area = .5*np.sum(inner[:, 0]*nxt[:, 1]-inner[:, 1]*nxt[:, 0])
    polygon_area = float(rectangle_area-hole_area)
    sagitta = radius-np.linalg.norm((inner+nxt)/2, axis=1).min()
    h_max = float(np.max(np.linalg.norm(vertices - np.roll(vertices, 1, axis=1), axis=2)))
    b = solution.reduced.boundary
    metrics = {"case": case.problem.name, "nr": nr, "nside": nside,
               "nodes": len(mesh.nodes), "triangles": len(mesh.triangles), "h_max": h_max,
               "minimum_triangle_area": float(areas.min()), "mesh_area": area,
               "expected_polygon_area": polygon_area, "circular_area": circular_area,
               "relative_geometry_area_error": abs(area-circular_area)/circular_area,
               "cylinder_boundary_sagitta": float(sagitta),
               "dirichlet_nodes": len(b), "neumann_edges": len(solution.assembly.boundary.neumann_edges),
               "max_boundary_error": float(np.max(np.abs(solution.values[b]-exact[b]))),
               "absolute_residual": solution.absolute_residual,
               "relative_residual": solution.relative_residual, **errors}
    return field, metrics


def run_experiment(*, nr: int = 8, nside: int | None = None, side_ratio: int = 2,
                   radius: float = .3, bounds=(-1., 1., -.75, .75), center=(0., 0.),
                   grading: float = 1., case: str = "quadratic", diffusivity: float = 1.,
                   boundary_mode: str = "dirichlet", convergence: list[int] | None = None,
                   quadrature_order: int = 4, error_quadrature_order: int = 5,
                   output: str | Path | None = None) -> dict:
    """Refine radial layers and every rectangle side together; preserve prior runs."""
    started = perf_counter()
    if isinstance(side_ratio, bool) or not isinstance(side_ratio, int) or side_ratio < 1:
        raise ValueError("side_ratio must be an integer >= 1")
    nside = side_ratio*nr if nside is None else nside
    if convergence is not None:
        if (len(convergence) < 2
                or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in convergence)
                or any(b <= a for a, b in zip(convergence, convergence[1:]))):
            raise ValueError("convergence sizes must be strictly increasing positive integers")
    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = Path("outputs/cylinder") / f"{boundary_mode}_{stamp}_{uuid4().hex[:8]}"
    dest = Path(output)
    if dest.exists() and (not dest.is_dir() or any(dest.iterdir())):
        raise ValueError(f"output directory must be empty to preserve previous runs: {dest}")
    settings = {"radius": radius, "bounds": list(bounds), "center": list(center),
                "grading": grading, "case": case, "diffusivity": diffusivity,
                "boundary_mode": boundary_mode, "quadrature_order": quadrature_order,
                "error_quadrature_order": error_quadrature_order}
    field, metrics = evaluate(nr, nside, **settings)
    dest.mkdir(parents=True, exist_ok=True)
    save_field(dest / "field.npz", field)
    plot_comparison(field, dest / "comparison.png")
    (dest / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    rows = []
    for n in convergence or []:
        row = evaluate(n, side_ratio*n, **settings)[1]
        for key, rate in (("relative_l2", "l2_rate"), ("relative_h1_seminorm", "h1_rate"),
                          ("relative_geometry_area_error", "geometry_area_rate")):
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
    config = {**settings, "phase": "2A-geometry", "geometry": "rectangle_with_cylinder",
              "case": case, "manufactured_case": field.case, "nr": nr, "nside": nside, "side_ratio": side_ratio,
              "convergence": convergence, "output": str(dest.resolve()), "random_seed": None,
              "error_integration_domain": "polygonal mesh; polynomial reference defined throughout plane",
              "neumann_convention": "k grad(u) dot actual outward facet normal"}
    return write_manifest(dest / "manifest.json", config,
                          {"single_run": metrics, "convergence": rows}, perf_counter()-started)
