"""CPU P2/P1 Stokes with full velocity or tagged velocity/traction conditions.

Unknown ordering is [ux at P2 nodes, uy at P2 nodes, p at vertices, optional gauge].
The symmetric saddle system uses -int(q div u) in its continuity row.
No pressure penalty, neural field or artificial compressibility is used.
"""
from dataclasses import dataclass
import warnings
import numpy as np
from scipy.sparse import coo_matrix, bmat, csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve, MatrixRankWarning
from physgraph.core.types import Mesh2D
from physgraph.fem.geometry import element_geometry
from physgraph.fem.quadrature import triangle_rule, interval_rule
from physgraph.fem.stokes_boundary import stokes_partition, facet_barycentric, stokes_velocity_values


def p2_basis(bary, gradients):
    """Vertex basis followed by edge (0,1), (1,2), (2,0) basis."""
    phi = np.column_stack([bary[:, i]*(2*bary[:, i]-1) for i in range(3)] +
                          [4*bary[:, i]*bary[:, j] for i, j in ((0, 1), (1, 2), (2, 0))])
    grad = np.stack([(4*bary[:, i]-1)[:, None]*gradients[i] for i in range(3)] +
                    [4*(bary[:, i, None]*gradients[j]+bary[:, j, None]*gradients[i])
                     for i, j in ((0, 1), (1, 2), (2, 0))], axis=1)
    return phi, grad


@dataclass
class TaylorHoodSpace:
    mesh: Mesh2D
    points: np.ndarray
    cells: np.ndarray
    boundary: np.ndarray


def taylor_hood_space(mesh):
    for tri in mesh.triangles:
        element_geometry(mesh.nodes[tri])
    local_edges = mesh.triangles[:, [[0, 1], [1, 2], [2, 0]]]
    edges, inverse, counts = np.unique(np.sort(local_edges.reshape(-1, 2), axis=1),
                                       axis=0, return_inverse=True, return_counts=True)
    if np.any(counts > 2):
        raise ValueError('nonmanifold triangle edges')
    exterior = edges[counts == 1]
    marked = np.zeros(len(mesh.nodes), bool)
    marked[np.unique(exterior)] = True
    if not np.array_equal(marked, mesh.boundary):
        raise ValueError('boundary mask must match all exterior edges')
    adjacency = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                           shape=(len(mesh.nodes),)*2).tocsr()
    if connected_components(adjacency, directed=False, return_labels=False) != 1:
        raise ValueError('Stokes requires a connected domain with one pressure gauge')
    if mesh.boundary_edges is not None:
        if set(map(tuple, np.sort(mesh.boundary_edges, axis=1))) != set(map(tuple, exterior)):
            raise ValueError('boundary tags must cover exactly the exterior edges')
    points = np.vstack((mesh.nodes, mesh.nodes[edges].mean(axis=1)))
    cells = np.column_stack((mesh.triangles, len(mesh.nodes)+inverse.reshape(-1, 3)))
    return TaylorHoodSpace(mesh, points, cells, np.r_[marked, counts == 1])


@dataclass
class StokesResult:
    space: TaylorHoodSpace
    velocity: np.ndarray
    pressure: np.ndarray
    diagnostics: dict
    viscosity: float = 1.
    viscous_form: str = 'laplacian'
    dirichlet_tags: tuple | None = None
    velocity_reactions: np.ndarray | None = None


def solve_stokes(mesh, forcing, boundary_velocity, *, viscosity=1., quadrature_order=6,
                 viscous_form='laplacian', dirichlet_tags=None, tractions=None):
    """Solve with explicit viscous form and an exhaustive boundary partition.

    Default ``laplacian`` preserves the earlier reference; ``symmetric_gradient``
    uses physical Newtonian stress. Traction callbacks have signature (x, normals).
    With mixed conditions, all non-Dirichlet tags require an explicit callback,
    including zero traction. With full velocity conditions, pressure has mean zero;
    otherwise prescribed traction sets its level without a pressure constraint.
    """
    if not np.isfinite(viscosity) or viscosity <= 0:
        raise ValueError('viscosity must be finite and positive')
    if isinstance(quadrature_order, bool) or not isinstance(quadrature_order, (int, np.integer)) or quadrature_order < 3:
        raise ValueError('Stokes quadrature_order must be an integer >= 3')
    if viscous_form not in ('laplacian', 'symmetric_gradient'):
        raise ValueError('unknown viscous form')
    space = taylor_hood_space(mesh)
    prescribed, facets, tractions, dirichlet_tags = stokes_partition(space, dirichlet_tags, tractions)
    mean_gauge = not tractions
    nv, npres = len(space.points), len(mesh.nodes)
    bary, weights = triangle_rule(quadrature_order)
    kr, kc, kv, br, bc, bv = [], [], [], [], [], []
    load = np.zeros(2*nv)
    mass = np.zeros(npres)
    for tri, dofs in zip(mesh.triangles, space.cells):
        vertices = mesh.nodes[tri]
        area, gradients = element_geometry(vertices)
        phi, grad = p2_basis(bary, gradients)
        w = area*weights
        force = np.asarray(forcing(bary @ vertices), dtype=float)
        if force.shape != (len(w), 2) or not np.isfinite(force).all():
            raise ValueError('forcing must return finite (N,2) values')
        local_k = viscosity*np.einsum('q,qid,qjd->ij', w, grad, grad)
        for i in range(2):
            for j in range(2):
                if i != j and viscous_form == 'laplacian':
                    continue
                block = local_k.copy() if i == j else np.zeros((6, 6))
                if viscous_form == 'symmetric_gradient':
                    block += viscosity*np.einsum('q,qi,qj->ij', w, grad[:, :, j], grad[:, :, i])
                kr.extend(np.repeat(dofs+i*nv, 6)); kc.extend(np.tile(dofs+j*nv, 6)); kv.extend(block.ravel())
        for component in range(2):
            local_b = -np.einsum('q,qi,qj->ij', w, bary, grad[:, :, component])
            br.extend(np.repeat(tri, 6)); bc.extend(np.tile(dofs+component*nv, 3)); bv.extend(local_b.ravel())
            load[dofs+component*nv] += np.einsum('q,qi,q->i', w, phi, force[:, component])
        mass[tri] += area/3
    body_force = load.reshape(2, nv).sum(axis=1).copy()
    traction_force = np.zeros(2)
    edge_points, edge_weights = interval_rule(quadrature_order)
    for facet in facets:
        if facet.tag not in tractions:
            continue
        tri, dofs = mesh.triangles[facet.cell], space.cells[facet.cell]
        edge_bary = facet_barycentric(facet, edge_points)
        _, gradients = element_geometry(mesh.nodes[tri])
        phi, _ = p2_basis(edge_bary, gradients)
        normals = np.broadcast_to(facet.normal, (len(edge_points), 2))
        traction = np.asarray(tractions[facet.tag](edge_bary @ mesh.nodes[tri], normals), dtype=float)
        if traction.shape != normals.shape or not np.isfinite(traction).all():
            raise ValueError('traction must return finite (N,2) values')
        integrated = facet.length*np.einsum('q,qi,qc->ic', edge_weights, phi, traction)
        load[dofs] += integrated[:, 0]
        load[dofs+nv] += integrated[:, 1]
        traction_force += facet.length*(edge_weights @ traction)
    k = coo_matrix((kv, (kr, kc)), shape=(2*nv, 2*nv)).tocsr()
    b = coo_matrix((bv, (br, bc)), shape=(npres, 2*nv)).tocsr()
    m = csr_matrix(mass[:, None])
    system = (bmat([[k, b.T, None], [b, None, m], [None, m.T, None]], format='csr')
              if mean_gauge else bmat([[k, b.T], [b, None]], format='csr'))
    rhs = np.r_[load, np.zeros(npres+int(mean_gauge))]
    values = stokes_velocity_values(space, prescribed, facets, dirichlet_tags, boundary_velocity)
    fixed = np.r_[prescribed, prescribed+nv]
    solution = np.zeros(len(rhs))
    solution[fixed] = values.T.ravel()
    # Sum of pressure test functions is one: this is the exact discrete net flux.
    flux = -float(np.sum(b @ solution[:2*nv]))
    if mean_gauge and abs(flux) > 1e-11*max(1., np.linalg.norm(values)):
        raise ValueError('incompatible prescribed boundary flux for incompressible flow')
    free = np.setdiff1d(np.arange(len(rhs)), fixed)
    reduced = system[free][:, free]
    lifted = (rhs-system @ solution)[free]
    with warnings.catch_warnings():
        warnings.simplefilter('error', MatrixRankWarning)
        solution[free] = spsolve(reduced, lifted)
    if not np.isfinite(solution).all():
        raise RuntimeError('nonfinite Stokes solution')
    residual = np.linalg.norm((system @ solution-rhs)[free])/max(np.linalg.norm(lifted), 1.)
    if residual > 1e-9:
        raise RuntimeError('Stokes linear residual failed')
    velocity = solution[:2*nv].reshape(2, nv).T.copy()
    pressure = solution[2*nv:2*nv+npres].copy()
    nodal_reactions = (system @ solution-rhs)[:2*nv].reshape(2, nv).T.copy()
    reactions = nodal_reactions[prescribed].sum(axis=0)
    diagnostics = dict(relative_residual=float(residual),
        pressure_mean=float(mass @ pressure/mass.sum()),
        pressure_constraint='mean_zero' if mean_gauge else 'traction',
        boundary_flux=-float(np.sum(b @ solution[:2*nv])),
        weak_divergence_residual=float(np.linalg.norm(b @ solution[:2*nv])),
        gauge_multiplier=float(solution[-1]) if mean_gauge else None,
        body_force=body_force.tolist(), prescribed_traction_force=traction_force.tolist(),
        dirichlet_reaction_on_fluid=reactions.tolist(),
        algebraic_force_balance=float(np.linalg.norm(body_force+traction_force+reactions)),
        max_boundary_error=float(np.max(np.abs(velocity[prescribed]-values))),
        velocity_nodes=nv, pressure_nodes=npres, triangles=len(mesh.triangles))
    return StokesResult(space, velocity, pressure, diagnostics, viscosity, viscous_form,
                        dirichlet_tags, nodal_reactions)
