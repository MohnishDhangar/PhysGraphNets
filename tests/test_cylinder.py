import json
from pathlib import Path
import subprocess
import sys

import matplotlib.tri as mtri
import numpy as np
import pytest

from physgraph.mesh.cylinder import cylinder_mesh
from physgraph.physics.cylinder import cylinder_case, BOUNDARY_MODES
from physgraph.experiments.cylinder import evaluate, run_experiment
from physgraph.fem.elliptic import solve
from physgraph.fem.boundary import exterior_edges
from physgraph.io.results import load_field


def test_rectangle_hole_topology_area_and_boundary_normals():
    center = np.array([.2, -.1])
    mesh = cylinder_mesh(3, 8, center=center, grading=1.3)
    assert len(np.unique(mesh.nodes, axis=0)) == len(mesh.nodes)
    edges = mesh.triangles[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    unique, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    assert set(counts) == {1, 2}
    assert len(mesh.nodes)-len(unique)+len(mesh.triangles) == 0
    assert len(exterior_edges(mesh)) == 64
    normals = {"bottom": [0, -1], "right": [1, 0], "top": [0, 1], "left": [-1, 0]}
    for edge, tag in zip(mesh.boundary_edges, mesh.boundary_edge_tags):
        a, b = mesh.nodes[edge]
        tangent = b-a
        normal = np.array([tangent[1], -tangent[0]])/np.linalg.norm(tangent)
        if tag == "cylinder":
            assert normal @ ((a+b)/2-center) < 0
        else:
            np.testing.assert_allclose(normal, normals[tag], atol=1e-14)
    finder = mtri.Triangulation(*mesh.nodes.T, triangles=mesh.triangles).get_trifinder()
    assert finder(*center) == -1
    assert finder(.9, .6) >= 0
    assert finder(1.1, 0) == -1
    _, row = evaluate(3, 8, center=center)
    assert row["minimum_triangle_area"] > 0
    assert row["mesh_area"] == pytest.approx(row["expected_polygon_area"], abs=1e-13)
    assert row["mesh_area"] > row["circular_area"]


@pytest.mark.parametrize("mode", BOUNDARY_MODES)
def test_affine_patch_and_nonzero_boundaries(mode):
    mesh = cylinder_mesh(4, 8, center=(.2, -.1), grading=1.3)
    case = cylinder_case("affine", mode, diffusivity=2.5)
    solution = solve(mesh, case.problem)
    np.testing.assert_allclose(solution.values, case.reference.exact(mesh.nodes), atol=2e-12, rtol=0)


@pytest.mark.parametrize("mode", BOUNDARY_MODES)
def test_quadratic_convergence(mode):
    rows = [evaluate(n, 2*n, boundary_mode=mode)[1] for n in (4, 8, 16)]
    for row in rows:
        assert row["relative_residual"] < 1e-10
        assert row["max_boundary_error"] < 1e-13
    for coarse, fine in zip(rows, rows[1:]):
        for key, low, high in (("relative_l2", 1.7, 2.4), ("relative_h1_seminorm", .8, 1.3),
                                ("relative_geometry_area_error", 1.7, 2.4)):
            rate = np.log(coarse[key]/fine[key])/np.log(coarse["h_max"]/fine["h_max"])
            assert low < rate < high, (mode, key, rate)


@pytest.mark.parametrize("kwargs", [{"nr": True}, {"nr": 0}, {"nside": 2.5},
    {"radius": 0}, {"radius": .75}, {"center": (1., 0)}, {"center": (np.nan, 0)},
    {"bounds": (1., -1., -.75, .75)}, {"bounds": (0., 1.)}, {"grading": 0}, {"grading": np.inf}])
def test_invalid_geometry(kwargs):
    with pytest.raises(ValueError):
        cylinder_mesh(**{"nr": 3, "nside": 8, **kwargs})


def test_cli_archive_and_provenance(tmp_path):
    dest = tmp_path / "run"
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"nr": 2, "boundary_mode": "outer_neumann"}))
    subprocess.run([sys.executable, "scripts/run_cylinder.py", "--config", str(config),
        "--nr", "3", "--convergence", "2", "4", "--output", str(dest)],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, check=True)
    manifest = json.loads((dest / "manifest.json").read_text())
    assert manifest["config"]["nr"] == 3
    assert manifest["config"]["nside"] == 6
    assert manifest["config"]["geometry"] == "rectangle_with_cylinder"
    assert manifest["python"] and len(manifest["source_sha256"]) == 64
    assert "code_revision" in manifest
    field, meta = load_field(dest / "field.npz")
    assert meta["schema_version"] == 2
    assert set(field.mesh.boundary_edge_tags) == {"cylinder", "bottom", "right", "top", "left"}
    for name in ("comparison.png", "metrics.json", "convergence.csv", "convergence.png"):
        assert (dest / name).stat().st_size > 0
    with pytest.raises(ValueError, match="preserve"):
        run_experiment(output=dest)
