"""Cached CPU Taylor-Hood operators for fully implicit transient flow.

Uses the verified Stokes space, basis and facet conventions. Velocity vectors
are component-major; local velocity arrays are (cell, local node, component).
"""
import numpy as np
from scipy.sparse import bmat, block_diag, coo_matrix, csr_matrix
from physgraph.fem.geometry import element_geometry
from physgraph.fem.quadrature import triangle_rule, interval_rule
from physgraph.fem.stokes import p2_basis
from physgraph.fem.stokes_boundary import facet_barycentric


class FlowOperators:
    def __init__(self, space, viscosity, quadrature_order=9):
        if not np.isfinite(viscosity) or viscosity <= 0:
            raise ValueError('viscosity must be finite and positive')
        if type(quadrature_order) is not int or quadrature_order < 4:
            raise ValueError('flow quadrature order must be an integer >= 4')
        self.space, self.viscosity = space, viscosity
        self.quadrature_order = quadrature_order
        mesh, cells = space.mesh, space.cells
        self.nv, self.np = len(space.points), len(mesh.nodes)
        bary, weights = triangle_rule(quadrature_order)
        geometry = [element_geometry(mesh.nodes[tri]) for tri in mesh.triangles]
        areas = np.array([g[0] for g in geometry])
        self.basis = p2_basis(bary, geometry[0][1])[0]
        self.grad = np.array([p2_basis(bary, g[1])[1] for g in geometry])
        self.weights = areas[:, None]*weights
        self.points = np.einsum('qi,tic->tqc', bary, mesh.nodes[mesh.triangles])
        self.rows = np.repeat(cells, 6, axis=1).ravel()
        self.cols = np.tile(cells, (1, 6)).ravel()
        local_mass = np.einsum('tq,qi,qj->tij', self.weights, self.basis, self.basis)
        self.mass_scalar = self.matrix(local_mass)
        self.mass = block_diag((self.mass_scalar, self.mass_scalar), format='csr')
        base = viscosity*np.einsum('tq,tqid,tqjd->tij', self.weights, self.grad, self.grad)
        blocks = [[self.matrix((base if i == j else 0.) + viscosity*np.einsum(
            'tq,tqi,tqj->tij', self.weights, self.grad[:, :, :, j], self.grad[:, :, :, i]))
            for j in range(2)] for i in range(2)]
        self.viscous = bmat(blocks, format='csr')
        br = np.repeat(mesh.triangles, 6, axis=1).ravel()
        bc = np.tile(cells, (1, 3)).ravel()
        divergence = []
        for component in range(2):
            values = -np.einsum('tq,qi,tqj->tij', self.weights, bary, self.grad[:, :, :, component])
            divergence.append(coo_matrix((values.ravel(), (br, bc)), shape=(self.np, self.nv)).tocsr())
        self.divergence = bmat([divergence], format='csr')
        self.pressure_integrals = np.zeros(self.np)
        np.add.at(self.pressure_integrals, mesh.triangles.ravel(), np.repeat(areas/3, 3))

    def matrix(self, local):
        return coo_matrix((local.ravel(), (self.rows, self.cols)), shape=(self.nv, self.nv)).tocsr()

    def scatter(self, local):
        vector = np.zeros((self.nv, 2))
        np.add.at(vector, self.space.cells.ravel(), local.reshape(-1, 2))
        return vector.T.ravel()

    def volume_load(self, callback):
        points = self.points.reshape(-1, 2)
        values = np.asarray(callback(points), dtype=float)
        if values.shape != points.shape or not np.isfinite(values).all():
            raise ValueError('volume data must return finite (N,2) values')
        return self.scatter(np.einsum('tq,qi,tqc->tic', self.weights, self.basis, values.reshape(self.points.shape)))

    def traction_load(self, facets, callbacks):
        answer = np.zeros((self.nv, 2))
        points, weights = interval_rule(self.quadrature_order)
        mesh = self.space.mesh
        for facet in facets:
            if facet.tag not in callbacks:
                continue
            bary = facet_barycentric(facet, points)
            vertices = mesh.nodes[mesh.triangles[facet.cell]]
            phi = p2_basis(bary, element_geometry(vertices)[1])[0]
            normals = np.broadcast_to(facet.normal, (len(points), 2))
            values = np.asarray(callbacks[facet.tag](bary @ vertices, normals), dtype=float)
            if values.shape != normals.shape or not np.isfinite(values).all():
                raise ValueError('traction must return finite (N,2) values')
            answer[self.space.cells[facet.cell]] += facet.length*np.einsum('q,qi,qc->ic', weights, phi, values)
        return answer.T.ravel()

    def convection(self, velocity, *, jacobian=False):
        local = velocity.reshape(2, self.nv).T[self.space.cells]
        u = np.einsum('qi,tic->tqc', self.basis, local)
        du = np.einsum('tqid,tic->tqcd', self.grad, local)
        adv = np.einsum('tqcd,tqd->tqc', du, u)
        value = self.scatter(np.einsum('tq,qi,tqc->tic', self.weights, self.basis, adv))
        if not jacobian:
            return value
        # D[(u.grad)u] du = (u.grad)du + (du.grad)u; retain both terms.
        transport = np.einsum('tq,qi,tqd,tqjd->tij', self.weights, self.basis, u, self.grad)
        blocks = [[self.matrix((transport if i == j else 0.) + np.einsum(
            'tq,qi,qj,tq->tij', self.weights, self.basis, self.basis, du[:, :, i, j]))
            for j in range(2)] for i in range(2)]
        return value, bmat(blocks, format='csr')

    def saddle(self, velocity_block, mean_gauge):
        b = self.divergence
        if mean_gauge:
            m = csr_matrix(self.pressure_integrals[:, None])
            return bmat([[velocity_block, b.T, None], [b, None, m], [None, m.T, None]], format='csr')
        return bmat([[velocity_block, b.T], [b, None]], format='csr')
