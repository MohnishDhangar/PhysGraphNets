from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import matplotlib.tri as mtri
import numpy as np
import pytest

from physgraph.experiments.airfoil import evaluate, run_experiment
from physgraph.fem.assembly import assemble
from physgraph.fem.boundary import exterior_edges
from physgraph.fem.elliptic import solve
from physgraph.io.results import load_field, save_field
from physgraph.mesh.airfoil import airfoil_mesh, half_thickness, profile_area
from physgraph.physics.airfoil import airfoil_case, BOUNDARY_MODES


def test_profile_formula_area_and_closed_endpoints():
    assert profile_area() == pytest.approx(.081706, abs=1e-15)
    assert profile_area(.24, 2.) == pytest.approx(8*.081706, abs=1e-14)
    np.testing.assert_array_equal(half_thickness(np.array([0., 1.])), [0., 0.])
    mesh = airfoil_mesh(2, 128)
    inner = mesh.nodes[:128]
    x = inner[:, 0]+.5
    expected = .6*(.2969*np.sqrt(x)-.126*x-.3516*x**2+.2843*x**3-.1036*x**4)
    np.testing.assert_allclose(np.abs(inner[:, 1]), expected, atol=2e-15, rtol=0)
    np.testing.assert_array_equal(inner[[0, 64]], [[.5, 0.], [-.5, 0.]])
    np.testing.assert_allclose(inner[1:64, 1], -inner[:64:-1, 1], atol=1e-15)


@pytest.mark.parametrize("thickness", [.06, .12, .24])
def test_airfoil_topology_rotation_normals_and_empty_hole(thickness):
    nr, nt = 3, 64
    center = np.array([.3, -.4])
    angle = np.deg2rad(23.)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    mesh = airfoil_mesh(nr, nt, thickness=thickness, center=center,
                        rotation_degrees=23., grading=1.3)
    assert len(mesh.nodes) == (nr+1)*nt
    assert len(mesh.triangles) == 2*nr*nt
    assert len(np.unique(mesh.nodes, axis=0)) == len(mesh.nodes)
    edges = mesh.triangles[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    unique, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    assert set(counts) == {1, 2}
    assert len(mesh.nodes)-len(unique)+len(mesh.triangles) == 0
    assert len(exterior_edges(mesh)) == mesh.boundary.sum() == 2*nt
    v = mesh.nodes[mesh.triangles]
    e, f = v[:, 1]-v[:, 0], v[:, 2]-v[:, 0]
    areas = .5*(e[:, 0]*f[:, 1]-e[:, 1]*f[:, 0])
    assert areas.min() > 0
    for edge, tag in zip(mesh.boundary_edges, mesh.boundary_edge_tags):
        a, b = mesh.nodes[edge]
        tangent = b-a
        normal = np.array([tangent[1], -tangent[0]])/np.linalg.norm(tangent)
        relative_midpoint = (a+b)/2-center
        dot = normal @ relative_midpoint
        if tag == "farfield":
            assert dot > 0
        else:
            assert dot < 0
            body_midpoint = relative_midpoint @ rotation
            assert body_midpoint[1] > 0 if tag == "airfoil_upper" else body_midpoint[1] < 0
    finder = mtri.Triangulation(*mesh.nodes.T, triangles=mesh.triangles).get_trifinder()
    assert finder(*center) == -1
    assert finder(*(center + rotation @ [.2, 0.])) == -1
    assert finder(*(center + rotation @ [0., .3])) >= 0
    assert finder(*(center + [2.1, 0.])) == -1


@pytest.mark.parametrize("mode", BOUNDARY_MODES)
def test_rotated_scaled_affine_patch(mode):
    field, row = evaluate(3, 48, case="affine", boundary_mode=mode, chord=1.3,
                          thickness=.18, diffusivity=2.5, grading=1.3,
                          rotation_degrees=-17., center=(.2, -.3))
    np.testing.assert_allclose(field.solution, field.exact, atol=1e-11, rtol=0)
    assert row["max_boundary_error"] == 0
    assert row["relative_residual"] < 1e-11
    assert row["mesh_area"] == pytest.approx(row["expected_polygon_area"], abs=1e-13)


@pytest.mark.parametrize("mode", BOUNDARY_MODES)
def test_quadratic_and_both_geometry_boundaries_converge(mode):
    rows = [evaluate(n, 16*n, boundary_mode=mode)[1] for n in (4, 8, 16)]
    for row in rows:
        assert row["relative_residual"] < 1e-10
        assert row["max_boundary_error"] == 0
        assert row["mesh_area"] == pytest.approx(row["expected_polygon_area"], abs=1e-13)
        assert row["polygon_airfoil_area"] < row["analytic_airfoil_area"]
        assert row["polygon_farfield_area"] < row["analytic_farfield_area"]
    for coarse, fine in zip(rows, rows[1:]):
        for key, low, high in (("relative_l2", 1.7, 2.5),
                                ("relative_h1_seminorm", .8, 1.3),
                                ("relative_airfoil_area_error", 1.7, 2.5),
                                ("relative_farfield_area_error", 1.7, 2.5)):
            rate = np.log(coarse[key]/fine[key])/np.log(coarse["h_max"]/fine["h_max"])
            assert low < rate < high, (mode, key, rate)


@pytest.mark.parametrize("mode", ["airfoil_neumann", "farfield_neumann"])
def test_quadratic_integrated_flux_matches_divergence_theorem(mode):
    field, metrics = evaluate(2, 32, center=(.7, -.4), rotation_degrees=31.)
    problem = airfoil_case(boundary_mode=mode, diffusivity=2.5).problem
    # Isolate the boundary load; Delta(1+x²+y²)=4 on either enclosed polygon.
    system = assemble(field.mesh, replace(problem, forcing=lambda p: np.zeros(len(p))))
    area = -metrics["polygon_airfoil_area"] if mode == "airfoil_neumann" else metrics["polygon_farfield_area"]
    assert system.load.sum() == pytest.approx(4*2.5*area, rel=1e-12)


def test_archive_tags_survive_storage_and_edge_reordering(tmp_path):
    field, _ = evaluate(2, 32, boundary_mode="airfoil_neumann")
    save_field(tmp_path / "field.npz", field)
    recovered, metadata = load_field(tmp_path / "field.npz")
    assert metadata["schema_version"] == 2
    mesh = recovered.mesh
    reordered = replace(mesh, boundary_edges=mesh.boundary_edges[::-1, ::-1],
                         boundary_edge_tags=mesh.boundary_edge_tags[::-1])
    problem = airfoil_case(boundary_mode="airfoil_neumann").problem
    np.testing.assert_array_equal(solve(reordered, problem).values, field.solution)


@pytest.mark.parametrize("kwargs", [
    {"nr": 0}, {"nr": True}, {"ntheta": 12}, {"ntheta": 18}, {"ntheta": 32.5},
    {"thickness": .01}, {"thickness": .3}, {"thickness": np.nan}, {"chord": 0},
    {"farfield_radius": .5}, {"center": [0.]}, {"center": [np.inf, 0]},
    {"rotation_degrees": np.inf}, {"grading": 0}, {"grading": 10000.},
    {"center": [1e20, 1e20]},
])
def test_invalid_or_collapsed_geometry(kwargs):
    with pytest.raises(ValueError):
        airfoil_mesh(**{"nr": 3, "ntheta": 32, **kwargs})


def test_invalid_profile_and_physics():
    for x in ([-.1], [1.1], [np.nan]):
        with pytest.raises(ValueError):
            half_thickness(x)
    for kwargs in ({"case": "flow"}, {"boundary_mode": "no_slip"}, {"diffusivity": 0}):
        with pytest.raises(ValueError):
            airfoil_case(**kwargs)


def test_cli_overrides_exports_and_provenance(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"nr": 2, "boundary_mode": "airfoil_neumann"}))
    dest = tmp_path / "run"
    subprocess.run([sys.executable, "scripts/run_airfoil.py", "--config", str(config),
                    "--nr", "3", "--convergence", "2", "4", "--output", str(dest)],
                   cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, check=True)
    manifest = json.loads((dest / "manifest.json").read_text())
    assert manifest["config"]["nr"] == 3
    assert manifest["config"]["ntheta"] == 48
    assert manifest["config"]["geometry"] == "symmetric_airfoil_exterior"
    assert manifest["config"]["boundary_mode"] == "airfoil_neumann"
    assert manifest["python"] and manifest["dependencies"]["numpy"]
    assert "code_revision" in manifest and len(manifest["source_sha256"]) == 64
    rows = manifest["metrics"]["convergence"]
    assert len(rows) == 2 and rows[-1]["airfoil_area_rate"] > 1.5
    for name in ("field.npz", "metrics.json", "comparison.png", "convergence.csv", "convergence.png"):
        assert (dest / name).stat().st_size > 0
    with pytest.raises(ValueError, match="preserve"):
        run_experiment(output=dest)


@pytest.mark.parametrize("kwargs", [{"angular_ratio": 10}, {"angular_ratio": True},
    {"convergence": [4, 2]}, {"convergence": [2]}, {"convergence": [2, 4.5]}])
def test_invalid_refinement_settings(tmp_path, kwargs):
    with pytest.raises(ValueError):
        run_experiment(output=tmp_path / "invalid", **kwargs)
