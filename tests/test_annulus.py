from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import matplotlib.tri as mtri
import numpy as np
import pytest

from physgraph.core.types import Mesh2D
from physgraph.experiments.annulus import evaluate, run_experiment
from physgraph.fem.assembly import assemble
from physgraph.fem.boundary import exterior_edges, partition_boundary
from physgraph.fem.elliptic import solve
from physgraph.fem.geometry import element_geometry
from physgraph.io.results import load_field, save_field
from physgraph.mesh.annulus import annulus_mesh
from physgraph.mesh.structured import unit_square_mesh
from physgraph.physics.annulus import BOUNDARY_MODES, radial_log_case
from physgraph.physics.base import EllipticProblem
from physgraph.physics.boundary import BoundaryConditions


def test_annulus_topology_area_normals_and_hole():
    nr, nt = 3, 16
    center = np.array([2., -1.])
    mesh = annulus_mesh(nr, nt, .5, 1., center, grading=1.3)
    assert len(mesh.nodes) == (nr+1)*nt
    assert len(mesh.triangles) == 2*nr*nt
    assert len(np.unique(mesh.nodes, axis=0)) == len(mesh.nodes)
    assert mesh.boundary.sum() == 2*nt
    areas = [element_geometry(mesh.nodes[t])[0] for t in mesh.triangles]
    assert min(areas) > 0
    assert sum(areas) == pytest.approx(.5*nt*np.sin(2*np.pi/nt)*(1-.5**2), abs=1e-13)
    edges = exterior_edges(mesh)
    assert len(edges) == 2*nt
    all_edges = mesh.triangles[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    unique, counts = np.unique(np.sort(all_edges, axis=1), axis=0, return_counts=True)
    assert set(counts) == {1, 2}
    assert len(mesh.nodes)-len(unique)+len(mesh.triangles) == 0  # One hole.
    tags = {tuple(sorted(edge)): tag for edge, tag in zip(mesh.boundary_edges, mesh.boundary_edge_tags)}
    for edge in edges:
        a, b = mesh.nodes[edge]
        tangent = b-a
        normal = np.array([tangent[1], -tangent[0]])/np.linalg.norm(tangent)
        dot = normal @ ((a+b)/2-center)
        assert dot < 0 if tags[tuple(sorted(edge))] == "inner" else dot > 0
    triangulation = mtri.Triangulation(*mesh.nodes.T, triangles=mesh.triangles)
    finder = triangulation.get_trifinder()
    assert finder(*center) == -1
    assert finder(*(center + [.2, 0])) == -1
    assert finder(*(center + [.75, 0])) >= 0
    assert finder(*(center + [1.1, 0])) == -1


@pytest.mark.parametrize("kwargs", [{"nr": 0}, {"ntheta": 2}, {"nr": True},
                                     {"nr": 2.5}, {"inner_radius": 0},
                                     {"inner_radius": 1}, {"outer_radius": np.inf},
                                     {"grading": 0}, {"center": (np.nan, 0)}])
def test_invalid_annulus_parameters(kwargs):
    settings = {"nr": 3, "ntheta": 24, **kwargs}
    with pytest.raises(ValueError):
        annulus_mesh(**settings)


@pytest.mark.parametrize("mode", BOUNDARY_MODES)
def test_log_case_convergence_and_geometry_refinement(mode):
    rows = [evaluate(n, 8*n, boundary_mode=mode)[1] for n in (4, 8, 16)]
    for row in rows:
        assert row["relative_residual"] < 1e-10
        assert row["max_boundary_error"] < 1e-13
        assert row["mesh_area"] == pytest.approx(row["expected_polygon_area"], abs=1e-13)
    for coarse, fine in zip(rows, rows[1:]):
        for key, low, high in (("relative_l2", 1.7, 2.3),
                                ("relative_h1_seminorm", .8, 1.2),
                                ("relative_geometry_area_error", 1.8, 2.2)):
            rate = np.log(coarse[key]/fine[key])/np.log(coarse["h_max"]/fine["h_max"])
            assert low < rate < high, (mode, key, rate)


@pytest.mark.parametrize("mode,sign", [("inner_neumann", -1), ("outer_neumann", 1)])
def test_integrated_neumann_load_has_correct_sign_and_magnitude(mode, sign):
    a, b, k, ui, uo = .4, 1.3, 2.5, -1., 3.
    center = (1.2, -.7)
    mesh = annulus_mesh(4, 64, a, b, center)
    case = radial_log_case(a, b, ui, uo, k, center, mode)
    system = assemble(mesh, case.problem)
    expected = sign*2*np.pi*k*(uo-ui)/np.log(b/a)
    assert system.load.sum() == pytest.approx(expected, rel=1e-10)
    assert len(system.boundary.dirichlet_nodes) == 64
    assert len(system.boundary.neumann_edges) == 64


@pytest.mark.parametrize("tag", ["inner", "outer"])
def test_nonradial_affine_patch_with_mixed_tagged_boundaries(tag):
    mesh = annulus_mesh(4, 32, center=(.2, -.3), grading=1.3)
    exact = lambda p: 1+2*p[:, 0]-3*p[:, 1]
    problem = EllipticProblem(
        "affine_annulus", lambda p: np.full(len(p), 2.), lambda p: np.ones(len(p)), exact,
        BoundaryConditions(exact, neumann=lambda p, n: 2*(2*n[:, 0]-3*n[:, 1]),
                           dirichlet_tags=(tag,)),
    )
    result = solve(mesh, problem)
    np.testing.assert_allclose(result.values, exact(mesh.nodes), atol=1e-11, rtol=0)


@pytest.mark.parametrize("mode", BOUNDARY_MODES)
def test_constant_reference_and_nonzero_boundary_values(mode):
    field, metrics = evaluate(3, 24, inner_value=2., outer_value=2., boundary_mode=mode)
    np.testing.assert_allclose(field.solution, 2., atol=1e-12, rtol=0)
    assert metrics["absolute_h1_seminorm"] < 1e-11


def test_tag_lookup_is_independent_of_storage_order_or_direction():
    mesh = annulus_mesh(3, 24)
    reordered = replace(mesh, boundary_edges=mesh.boundary_edges[::-1, ::-1],
                         boundary_edge_tags=mesh.boundary_edge_tags[::-1])
    case = radial_log_case(boundary_mode="inner_neumann")
    np.testing.assert_array_equal(solve(mesh, case.problem).values, solve(reordered, case.problem).values)


def test_tag_validation_and_missing_coverage():
    mesh = annulus_mesh(3, 24)
    with pytest.raises(ValueError, match="together"):
        replace(mesh, boundary_edge_tags=None)
    with pytest.raises(ValueError, match="unknown"):
        partition_boundary(mesh, BoundaryConditions(lambda p: np.zeros(len(p)), dirichlet_tags=("inlet",)))
    with pytest.raises(ValueError, match="either"):
        BoundaryConditions(lambda p: np.zeros(len(p)), dirichlet_selector=lambda p: np.ones(len(p), bool),
                           dirichlet_tags=("inner",))
    with pytest.raises(ValueError, match="requires a mesh"):
        partition_boundary(unit_square_mesh(3), radial_log_case().problem.boundary)
    incomplete = replace(mesh, boundary_edges=mesh.boundary_edges[1:],
                          boundary_edge_tags=mesh.boundary_edge_tags[1:])
    with pytest.raises(ValueError, match="every exterior"):
        solve(incomplete, radial_log_case().problem)


def test_tagged_archive_roundtrip_and_validation(tmp_path):
    field, _ = evaluate(3, 24, boundary_mode="inner_neumann", center=(.2, -.3))
    path = tmp_path / "field.npz"
    save_field(path, field)
    recovered, metadata = load_field(path)
    assert metadata["schema_version"] == 2
    np.testing.assert_array_equal(recovered.mesh.boundary_edges, field.mesh.boundary_edges)
    np.testing.assert_array_equal(recovered.mesh.boundary_edge_tags, field.mesh.boundary_edge_tags)
    case = radial_log_case(boundary_mode="inner_neumann", center=(.2, -.3))
    np.testing.assert_allclose(solve(recovered.mesh, case.problem).values, field.solution)
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name].copy() for name in archive.files}
    data["boundary_edges"] = data["boundary_edges"][1:]
    data["boundary_edge_tags"] = data["boundary_edge_tags"][1:]
    np.savez_compressed(tmp_path / "incomplete.npz", **data)
    with pytest.raises(ValueError, match="every exterior"):
        load_field(tmp_path / "incomplete.npz")
    del data["boundary_edge_tags"]
    np.savez_compressed(tmp_path / "missing.npz", **data)
    with pytest.raises(ValueError, match="requires boundary"):
        load_field(tmp_path / "missing.npz")


def test_annulus_cli_exports_geometry_and_provenance(tmp_path):
    config = tmp_path / "config.json"
    dest = tmp_path / "result"
    config.write_text(json.dumps({"nr": 2, "center": [.3, -.2],
                                  "boundary_mode": "inner_neumann", "output": str(dest)}))
    project = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, "scripts/run_annulus.py", "--config", str(config),
                    "--nr", "3", "--convergence", "2", "4"], cwd=project,
                   capture_output=True, text=True, check=True)
    manifest = json.loads((dest / "manifest.json").read_text())
    assert manifest["config"]["nr"] == 3
    assert manifest["config"]["ntheta"] == 24
    assert manifest["config"]["geometry"] == "polygonal_annulus"
    assert "code_revision" in manifest and len(manifest["source_sha256"]) == 64
    assert manifest["python"] and manifest["dependencies"]["numpy"]
    rows = manifest["metrics"]["convergence"]
    assert len(rows) == 2 and rows[-1]["l2_rate"] > 1.7
    for name in ("field.npz", "metrics.json", "comparison.png", "convergence.csv", "convergence.png"):
        assert (dest / name).stat().st_size > 0
    with pytest.raises(ValueError, match="preserve"):
        run_experiment(output=dest)
