"""Quality and topology audit for generated planar obstacle meshes.

This is not a general triangle-overlap detector or an arbitrary CAD importer.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from physgraph.fem.boundary import exterior_edges


def mesh_quality(mesh):
    xyz = mesh.nodes[mesh.triangles]
    a, b = xyz[:, 1]-xyz[:, 0], xyz[:, 2]-xyz[:, 0]
    cross = a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0]
    if not np.isfinite(cross).all() or np.any(cross <= 0):
        raise ValueError('mesh must have finite positive oriented triangle areas')
    lengths = np.linalg.norm(np.roll(xyz, -1, axis=1)-xyz, axis=2)
    angles = []
    for i in range(3):
        l, m, opposite = lengths[:, i], lengths[:, (i+2) % 3], lengths[:, (i+1) % 3]
        angles.append(np.arccos(np.clip((l*l+m*m-opposite*opposite)/(2*l*m), -1, 1)))
    quality = 2*np.sqrt(3)*cross/(lengths*lengths).sum(axis=1)
    all_edges = np.sort(mesh.triangles[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    unique = np.unique(all_edges, axis=0)
    graph = coo_matrix((np.ones(len(unique)), (unique[:, 0], unique[:, 1])), shape=(len(mesh.nodes),)*2).tocsr()
    if connected_components(graph, directed=False, return_labels=False) != 1:
        raise ValueError('mesh must be connected')
    edges = exterior_edges(mesh)
    starts, ends = edges.T
    if len(np.unique(starts)) != len(starts) or set(starts) != set(ends):
        raise ValueError('exterior boundaries must form disjoint simple cycles')
    successor = dict(zip(starts.tolist(), ends.tolist()))
    remaining, signed_areas = set(successor), []
    while remaining:
        start = min(remaining)
        loop, node = [], start
        while node in remaining:
            loop.append(node)
            remaining.remove(node)
            node = successor[node]
        if node != start:
            raise ValueError('invalid boundary cycle')
        xy = mesh.nodes[loop]-mesh.nodes[loop].mean(axis=0)
        following = np.roll(xy, -1, axis=0)
        signed_areas.append(float(.5*np.sum(xy[:, 0]*following[:, 1]-xy[:, 1]*following[:, 0])))
    if sum(area > 0 for area in signed_areas) != 1 or any(area == 0 for area in signed_areas):
        raise ValueError('require one positive outer boundary and negative hole boundaries')
    euler = len(mesh.nodes)-len(unique)+len(mesh.triangles)
    if euler != 2-len(signed_areas):
        raise ValueError('mesh Euler characteristic disagrees with boundary loops')
    area = float(cross.sum()/2)
    if not np.isclose(area, sum(signed_areas), rtol=1e-11, atol=1e-13):
        raise ValueError('triangle and oriented boundary areas disagree')
    tag_map = ({} if mesh.boundary_edges is None else
               {tuple(sorted(edge)): str(tag) for edge, tag in zip(mesh.boundary_edges, mesh.boundary_edge_tags)})
    boundary_sizes = {}
    for edge in edges:
        tag = tag_map.get(tuple(sorted(edge)), 'boundary')
        boundary_sizes.setdefault(tag, []).append(float(np.linalg.norm(mesh.nodes[edge[1]]-mesh.nodes[edge[0]])))
    boundary_summary = {tag: {'edges': len(values), 'min_edge': min(values),
                             'max_edge': max(values), 'length': sum(values)}
                        for tag, values in boundary_sizes.items()}
    return {'nodes': len(mesh.nodes), 'triangles': len(mesh.triangles), 'area': area,
            'min_area': float(cross.min()/2), 'min_angle_degrees': float(np.rad2deg(np.min(angles))),
            'min_scaled_quality': float(quality.min()), 'max_edge': float(lengths.max()),
            'boundary_loops': len(signed_areas), 'holes': len(signed_areas)-1,
            'boundary_signed_areas': sorted(signed_areas), 'euler_characteristic': euler,
            'boundary_by_tag': boundary_summary,
            'scope': 'generated-mesh topology and quality; global geometric overlap not certified'}
