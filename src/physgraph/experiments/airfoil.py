"""Airfoil exterior verification, separating profile, far-field and FEM errors."""
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
from physgraph.mesh.airfoil import airfoil_mesh, profile_area
from physgraph.physics.airfoil import airfoil_case
from physgraph.visualization.plotting import plot_comparison, plot_convergence


def evaluate(nr: int, ntheta: int, *, thickness: float = .12, chord: float = 1.,
             farfield_radius: float = 2., center=(0., 0.), rotation_degrees: float = 0.,
             grading: float = 1., case: str = "quadratic", diffusivity: float = 1.,
             boundary_mode: str = "dirichlet", quadrature_order: int = 4,
             error_quadrature_order: int = 5) -> tuple[FieldResult, dict]:
    mesh = airfoil_mesh(nr, ntheta, thickness=thickness, chord=chord,
                        farfield_radius=farfield_radius, center=center,
                        rotation_degrees=rotation_degrees, grading=grading)
    case = airfoil_case(case, boundary_mode, diffusivity)
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
    exact_hole_area = profile_area(thickness, chord)
    exact_outer_area = float(np.pi*farfield_radius**2)
    ideal_area = exact_outer_area-exact_hole_area
    inner = mesh.nodes[:ntheta]-np.asarray(center)
    nxt = np.roll(inner, -1, axis=0)
    hole_area = float(.5*np.sum(inner[:, 0]*nxt[:, 1]-inner[:, 1]*nxt[:, 0]))
    outer_area = float(.5*ntheta*farfield_radius**2*np.sin(2*np.pi/ntheta))
    polygon_area = outer_area-hole_area
    h_max = float(np.max(np.linalg.norm(vertices - np.roll(vertices, 1, axis=1), axis=2)))
    b = solution.reduced.boundary
    metrics = {"case": case.problem.name, "nr": nr, "ntheta": ntheta,
               "nodes": len(mesh.nodes), "triangles": len(mesh.triangles), "h_max": h_max,
               "minimum_triangle_area": float(areas.min()), "mesh_area": area,
               "expected_polygon_area": polygon_area, "ideal_domain_area": ideal_area,
               "analytic_airfoil_area": exact_hole_area, "polygon_airfoil_area": hole_area,
               "analytic_farfield_area": exact_outer_area, "polygon_farfield_area": outer_area,
               "relative_airfoil_area_error": abs(hole_area-exact_hole_area)/exact_hole_area,
               "relative_farfield_area_error": abs(outer_area-exact_outer_area)/exact_outer_area,
               "relative_geometry_area_error": abs(area-ideal_area)/ideal_area,
               "farfield_sagitta": float(farfield_radius*(1-np.cos(np.pi/ntheta))),
               "dirichlet_nodes": len(b), "neumann_edges": len(solution.assembly.boundary.neumann_edges),
               "max_boundary_error": float(np.max(np.abs(solution.values[b]-exact[b]))),
               "absolute_residual": solution.absolute_residual,
               "relative_residual": solution.relative_residual, **errors}
    return field, metrics


def run_experiment(*, nr: int = 8, ntheta: int | None = None, angular_ratio: int = 16,
                   thickness: float = .12, chord: float = 1., farfield_radius: float = 2.,
                   center=(0., 0.), rotation_degrees: float = 0., grading: float = 1.,
                   case: str = "quadratic", diffusivity: float = 1.,
                   boundary_mode: str = "dirichlet", convergence: list[int] | None = None,
                   quadrature_order: int = 4, error_quadrature_order: int = 5,
                   output: str | Path | None = None) -> dict:
    """Refine radial and angular counts together; preserve existing artifacts."""
    started = perf_counter()
    if (isinstance(angular_ratio, bool) or not isinstance(angular_ratio, int)
            or angular_ratio < 16 or angular_ratio % 4):
        raise ValueError("angular_ratio must be an integer >= 16 divisible by 4")
    ntheta = angular_ratio*nr if ntheta is None else ntheta
    if convergence is not None:
        if (len(convergence) < 2
                or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in convergence)
                or any(b <= a for a, b in zip(convergence, convergence[1:]))):
            raise ValueError("convergence sizes must be strictly increasing positive integers")
    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = Path("outputs/airfoil") / f"{boundary_mode}_{stamp}_{uuid4().hex[:8]}"
    dest = Path(output)
    if dest.exists() and (not dest.is_dir() or any(dest.iterdir())):
        raise ValueError(f"output directory must be empty to preserve previous runs: {dest}")
    settings = {"thickness": thickness, "chord": chord, "farfield_radius": farfield_radius,
                "center": list(center), "rotation_degrees": rotation_degrees,
                "grading": grading, "case": case, "diffusivity": diffusivity,
                "boundary_mode": boundary_mode, "quadrature_order": quadrature_order,
                "error_quadrature_order": error_quadrature_order}
    field, metrics = evaluate(nr, ntheta, **settings)
    dest.mkdir(parents=True, exist_ok=True)
    save_field(dest / "field.npz", field)
    plot_comparison(field, dest / "comparison.png")
    (dest / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    rows = []
    for n in convergence or []:
        row = evaluate(n, angular_ratio*n, **settings)[1]
        for key, rate in (("relative_l2", "l2_rate"), ("relative_h1_seminorm", "h1_rate"),
                          ("relative_airfoil_area_error", "airfoil_area_rate"),
                          ("relative_farfield_area_error", "farfield_area_rate")):
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
    config = {**settings, "phase": "2A-geometry", "geometry": "symmetric_airfoil_exterior",
              "profile": "closed symmetric NACA-style; x^4 coefficient -0.1036",
              "manufactured_case": field.case, "nr": nr, "ntheta": ntheta, "angular_ratio": angular_ratio,
              "convergence": convergence, "output": str(dest.resolve()), "random_seed": None,
              "error_integration_domain": "polygonal mesh; polynomial reference defined throughout plane",
              "neumann_convention": "k grad(u) dot actual outward facet normal"}
    return write_manifest(dest / "manifest.json", config,
                          {"single_run": metrics, "convergence": rows}, perf_counter()-started)
