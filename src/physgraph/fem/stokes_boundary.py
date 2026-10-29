"""Exterior facet geometry and explicit velocity/traction boundary partitions."""
from dataclasses import dataclass
from collections.abc import Mapping
import numpy as np
from physgraph.fem.boundary import exterior_edges


@dataclass(frozen=True)
class StokesFacet:
    cell: int
    local_edge: int
    vertices: tuple
    tag: str
    normal: np.ndarray
    length: float


def stokes_facets(mesh):
    edges = exterior_edges(mesh)
    owners = {}
    for cell, tri in enumerate(mesh.triangles):
        for local, (i, j) in enumerate(((0, 1), (1, 2), (2, 0))):
            owners[tuple(sorted((tri[i], tri[j])))] = (cell, local)
    tags = ({} if mesh.boundary_edges is None else
            {tuple(sorted(edge)): str(tag) for edge, tag in zip(mesh.boundary_edges, mesh.boundary_edge_tags)})
    facets = []
    for edge in edges:
        key = tuple(sorted(edge))
        delta = mesh.nodes[edge[1]]-mesh.nodes[edge[0]]
        length = float(np.linalg.norm(delta))
        if length <= 0 or not np.isfinite(length):
            raise ValueError('degenerate boundary edge')
        normal = np.array([delta[1], -delta[0]])/length
        cell, local = owners[key]
        facets.append(StokesFacet(cell, local, tuple(edge), tags.get(key, 'boundary'), normal, length))
    return facets


def stokes_partition(space, dirichlet_tags, tractions):
    facets = stokes_facets(space.mesh)
    if dirichlet_tags is None:
        if tractions is not None and len(tractions):
            raise ValueError('traction data require explicit Dirichlet tags')
        return np.flatnonzero(space.boundary), facets, {}, None
    if space.mesh.boundary_edges is None:
        raise ValueError('mixed conditions require tagged boundary edges')
    if isinstance(dirichlet_tags, str):
        raise ValueError('Dirichlet tags must be a collection of tag names')
    selected = set(dirichlet_tags)
    available = {facet.tag for facet in facets}
    if not selected or not selected <= available:
        raise ValueError('require nonempty known Dirichlet tags; pure traction is unsupported')
    if tractions is None:
        tractions = {}
    if not isinstance(tractions, Mapping) or set(tractions) != available-selected:
        raise ValueError('traction tags must cover exactly the non-Dirichlet boundary')
    if not all(callable(f) for f in tractions.values()):
        raise ValueError('each traction must be callable(points, outward_normals)')
    dofs = []
    for facet in facets:
        if facet.tag in selected:
            dofs.extend(facet.vertices)
            dofs.append(space.cells[facet.cell, 3+facet.local_edge])
    return np.unique(dofs), facets, dict(tractions), tuple(sorted(selected))


def facet_barycentric(facet, points):
    """Interval coordinates follow the owning CCW triangle, independent of tags."""
    i, j = ((0, 1), (1, 2), (2, 0))[facet.local_edge]
    bary = np.zeros((len(points), 3))
    bary[:, i], bary[:, j] = 1-points, points
    return bary


def stokes_velocity_values(space, prescribed, facets, dirichlet_tags, data):
    """Resolve callable or per-tag velocity data, rejecting conflicting corners."""
    if callable(data):
        values = np.asarray(data(space.points[prescribed]), dtype=float)
    else:
        selected = {f.tag for f in facets} if dirichlet_tags is None else set(dirichlet_tags)
        if not isinstance(data, Mapping) or set(data) != selected or not all(callable(v) for v in data.values()):
            raise ValueError('velocity mapping must cover exactly the Dirichlet tags')
        full = np.full((len(space.points), 2), np.nan)
        for tag in sorted(selected):
            dofs = []
            for facet in facets:
                if facet.tag == tag:
                    dofs.extend(facet.vertices)
                    dofs.append(space.cells[facet.cell, 3+facet.local_edge])
            dofs = np.unique(dofs)
            assigned = np.asarray(data[tag](space.points[dofs]), dtype=float)
            if assigned.shape != (len(dofs), 2) or not np.isfinite(assigned).all():
                raise ValueError('boundary velocity must return finite (N,2) values')
            shared = np.isfinite(full[dofs]).all(axis=1)
            if not np.allclose(full[dofs[shared]], assigned[shared], rtol=1e-12, atol=1e-12):
                raise ValueError('conflicting velocity values at shared Dirichlet corners')
            full[dofs] = assigned
        values = full[prescribed]
    if values.shape != (len(prescribed), 2) or not np.isfinite(values).all():
        raise ValueError('boundary velocity must return finite (N,2) values')
    return values
