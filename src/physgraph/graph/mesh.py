"""Directed mesh graphs with geometric features and independently supplied physics."""
from dataclasses import dataclass
import numpy as np
from physgraph.fem.boundary import exterior_edges, partition_boundary
from physgraph.fem.geometry import element_geometry
from physgraph.physics.base import sample_scalar


def frozen_array(values, dtype=None):
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class MeshGraph:
    nodes: np.ndarray
    edge_index: np.ndarray
    displacement: np.ndarray
    length: np.ndarray
    node_area: np.ndarray
    boundary: np.ndarray
    edge_boundary: np.ndarray
    edge_tags: np.ndarray
    outward_normal: np.ndarray


def build_graph(mesh):
    """Two directed arcs per triangle edge; no self loops or proximity edges.

    Displacement is target minus source. Outward normals belong to physical
    exterior facets, so BOTH graph directions carry the SAME outward normal.
    """
    exterior = exterior_edges(mesh)
    areas = np.array([element_geometry(mesh.nodes[t])[0] for t in mesh.triangles])
    node_area = np.bincount(mesh.triangles.ravel(), weights=np.repeat(areas/3, 3), minlength=len(mesh.nodes))
    undirected = np.unique(np.sort(mesh.triangles[:, [[0,1],[1,2],[2,0]]].reshape(-1,2), axis=1), axis=0)
    edges = np.vstack((undirected, undirected[:, ::-1]))
    edges = edges[np.lexsort((edges[:,1], edges[:,0]))]
    displacement = mesh.nodes[edges[:,1]]-mesh.nodes[edges[:,0]]
    length = np.linalg.norm(displacement, axis=1)
    if np.any(length <= 0) or not np.isfinite(length).all():
        raise ValueError('graph edges must have positive finite length')
    tag_map = ({tuple(sorted(e)): str(tag) for e,tag in zip(mesh.boundary_edges,mesh.boundary_edge_tags)}
               if mesh.boundary_edges is not None else {})
    normal_map = {}
    for a,b in exterior:
        dx,dy = mesh.nodes[b]-mesh.nodes[a]
        normal_map[tuple(sorted((a,b)))] = np.array([dy,-dx])/np.hypot(dx,dy)
    edge_boundary = np.array([tuple(sorted(e)) in normal_map for e in edges])
    normals = np.array([normal_map.get(tuple(sorted(e)), np.zeros(2)) for e in edges])
    tags = np.array([tag_map.get(tuple(sorted(e)), 'boundary' if boundary else '')
                     for e,boundary in zip(edges,edge_boundary)], dtype=str)
    return MeshGraph(*(frozen_array(a) for a in (mesh.nodes, edges.T, displacement, length,
                     node_area, mesh.boundary, edge_boundary, tags, normals)))


def physical_features(mesh, graph, problem):
    """Return physical inputs only; never read analytical or FEM solution fields."""
    if not np.array_equal(mesh.nodes, graph.nodes):
        raise ValueError('graph and physical mesh coordinates must match')
    partition = partition_boundary(mesh, problem.boundary)
    mask = np.zeros(len(mesh.nodes), dtype=bool)
    mask[partition.dirichlet_nodes] = True
    values = np.zeros(len(mesh.nodes))
    values[mask] = sample_scalar(problem.boundary.dirichlet, mesh.nodes[mask], 'Dirichlet data')
    neumann_edges = {tuple(sorted(e)) for e in partition.neumann_edges}
    neumann_mask = np.array([tuple(sorted(e)) in neumann_edges for e in graph.edge_index.T])
    flux = np.zeros(len(graph.length))
    if np.any(neumann_mask):
        edges = graph.edge_index[:, neumann_mask]
        midpoints = (graph.nodes[edges[0]]+graph.nodes[edges[1]])/2
        normals = graph.outward_normal[neumann_mask]
        flux[neumann_mask] = sample_scalar(lambda p: problem.boundary.neumann(p,normals), midpoints, 'Neumann data')
    coefficients = np.column_stack([sample_scalar(f,mesh.nodes,name) for f,name in
                    ((problem.diffusion,'diffusion'),(problem.reaction,'reaction'),(problem.forcing,'forcing'))])
    if np.any(coefficients[:,0] <= 0) or np.any(coefficients[:,1] < 0):
        raise ValueError('graph physics requires positive diffusion and nonnegative reaction')
    return {'coefficients': coefficients, 'dirichlet_mask': mask, 'dirichlet_values': values,
            'neumann_mask': neumann_mask, 'neumann_flux': flux}
