"""Exterior edge topology and Dirichlet/Neumann partitioning."""
from dataclasses import dataclass

import numpy as np

from physgraph.core.types import Mesh2D
from physgraph.physics.boundary import BoundaryConditions


@dataclass(frozen=True)
class BoundaryPartition:
    dirichlet_nodes: np.ndarray
    neumann_edges: np.ndarray


def exterior_edges(mesh: Mesh2D) -> np.ndarray:
    """Return oriented exterior edges; CCW triangles put the domain on the left."""
    edges = mesh.triangles[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    canonical = np.sort(edges, axis=1)
    _, first, inverse, counts = np.unique(canonical, axis=0, return_index=True,
                                         return_inverse=True, return_counts=True)
    if np.any(counts > 2):
        raise ValueError("nonmanifold edge belongs to more than two triangles")
    directions = np.where(edges[:, 0] < edges[:, 1], 1, -1)
    sums = np.bincount(inverse, weights=directions)
    if np.any((counts == 2) & (sums != 0)):
        raise ValueError("adjacent triangles have inconsistent edge orientation")
    exterior = edges[first[counts == 1]]
    actual_boundary = np.zeros(len(mesh.nodes), dtype=bool)
    actual_boundary[exterior.ravel()] = True
    if not np.array_equal(actual_boundary, mesh.boundary):
        raise ValueError("boundary mask does not match exterior mesh edges")
    if mesh.boundary_edges is not None:
        tagged = np.unique(np.sort(mesh.boundary_edges, axis=1), axis=0)
        expected = np.unique(np.sort(exterior, axis=1), axis=0)
        if not np.array_equal(tagged, expected):
            raise ValueError("tagged edges must cover every exterior edge exactly once")
    return exterior


def partition_boundary(mesh: Mesh2D, conditions: BoundaryConditions) -> BoundaryPartition:
    edges = exterior_edges(mesh)
    if conditions.dirichlet_tags is not None:
        if mesh.boundary_edges is None:
            raise ValueError("dirichlet_tags requires a mesh with tagged boundary edges")
        unknown = set(conditions.dirichlet_tags) - set(mesh.boundary_edge_tags)
        if unknown:
            raise ValueError(f"unknown Dirichlet boundary tags: {sorted(unknown)}")
        tags_by_edge = {tuple(sorted(edge)): tag
                        for edge, tag in zip(mesh.boundary_edges, mesh.boundary_edge_tags)}
        selected = np.array([tags_by_edge[tuple(sorted(edge))] in conditions.dirichlet_tags
                             for edge in edges], dtype=bool)
    elif conditions.dirichlet_selector is None:
        selected = np.ones(len(edges), dtype=bool)
    else:
        midpoints = mesh.nodes[edges].mean(axis=1)
        selected = np.asarray(conditions.dirichlet_selector(midpoints))
        if selected.shape != (len(edges),) or selected.dtype != np.bool_:
            raise ValueError("dirichlet_selector must return one boolean per boundary edge")
    if not np.any(selected):
        raise ValueError("Phase 2A requires a nonempty Dirichlet boundary; pure Neumann is unsupported")
    if np.any(~selected) and conditions.neumann is None:
        raise ValueError("a Neumann function is required for all non-Dirichlet exterior edges")
    return BoundaryPartition(np.unique(edges[selected]), edges[~selected])
