"""Quadrature errors of the P2 velocity and P1 pressure; no nodal proxies."""
import numpy as np
from physgraph.fem.geometry import element_geometry
from physgraph.fem.quadrature import triangle_rule, interval_rule
from physgraph.fem.stokes import p2_basis
from physgraph.fem.stokes_boundary import stokes_facets, facet_barycentric


def stokes_errors(result, case, quadrature_order=9):
    bary, weights = triangle_rule(quadrature_order)
    sums = np.zeros(7)
    mesh = result.space.mesh
    for tri, dofs in zip(mesh.triangles, result.space.cells):
        vertices = mesh.nodes[tri]
        area, gradients = element_geometry(vertices)
        phi, grad = p2_basis(bary, gradients)
        uh = phi @ result.velocity[dofs]
        duh = np.einsum('qjd,jc->qcd', grad, result.velocity[dofs])
        ph = bary @ result.pressure[tri]
        u, du, p, _ = case.fields(bary @ vertices)
        fields = [np.sum((uh-u)**2, axis=1), np.sum((duh-du)**2, axis=(1, 2)),
                  (ph-p)**2, np.trace(duh, axis1=1, axis2=2)**2,
                  np.sum(u*u, axis=1), np.sum(du*du, axis=(1, 2)), p*p]
        sums += area*np.array([weights @ field for field in fields])
    errors = dict(zip(('velocity_l2', 'velocity_h1_seminorm', 'pressure_l2', 'divergence_l2'),
                      map(float, np.sqrt(np.maximum(sums[:4], 0.)))))
    for i, name in enumerate(('velocity_l2', 'velocity_h1_seminorm', 'pressure_l2')):
        errors['relative_'+name] = float(np.sqrt(sums[i]/max(sums[4+i], 1e-30)))
    return errors


def stokes_boundary_integrals(result, case=None, quadrature_order=9):
    """Integrate flux and stress on each fluid boundary, including hole normals.

    Symmetric-gradient loads are physical stress*n on fluid, per unit density/depth.
    Laplacian loads are explicitly labeled conormal fluxes. Reconstructed balance
    is a separate refinement diagnostic, not the algebraic reaction.
    """
    points, weights = interval_rule(quadrature_order)
    mesh = result.space.mesh
    groups = {}
    for facet in stokes_facets(mesh):
        tri, dofs = mesh.triangles[facet.cell], result.space.cells[facet.cell]
        bary = facet_barycentric(facet, points)
        _, gradients = element_geometry(mesh.nodes[tri])
        phi, grad = p2_basis(bary, gradients)
        velocity = phi @ result.velocity[dofs]
        du = np.einsum('qjd,jc->qcd', grad, result.velocity[dofs])
        if result.viscous_form == 'symmetric_gradient':
            du = du + du.transpose(0, 2, 1)
        pressure = bary @ result.pressure[tri]
        normals = np.broadcast_to(facet.normal, (len(points), 2))
        stress_n = result.viscosity*np.einsum('qij,qj->qi', du, normals)-pressure[:, None]*normals
        group = groups.setdefault(facet.tag, {'length': 0., 'flux': 0., 'force_on_fluid': np.zeros(2),
                                             'exact_flux': 0., 'exact_force_on_fluid': np.zeros(2),
                                             'traction_squared_error': 0.})
        w = facet.length*weights
        group['length'] += facet.length
        group['flux'] += float(w @ np.sum(velocity*normals, axis=1))
        group['force_on_fluid'] += w @ stress_n
        if case is not None:
            positions = bary @ mesh.nodes[tri]
            exact_t = case.traction(positions, normals, result.viscous_form)
            group['exact_flux'] += float(w @ np.sum(case.velocity(positions)*normals, axis=1))
            group['exact_force_on_fluid'] += w @ exact_t
            group['traction_squared_error'] += float(w @ np.sum((stress_n-exact_t)**2, axis=1))
    net_force = sum((g['force_on_fluid'] for g in groups.values()), np.zeros(2))
    net_flux = sum(g['flux'] for g in groups.values())
    force_error = max(np.linalg.norm(g['force_on_fluid']-g['exact_force_on_fluid']) for g in groups.values())
    traction_error = np.sqrt(sum(g['traction_squared_error'] for g in groups.values()))
    for group in groups.values():
        group['force_on_fluid'] = group['force_on_fluid'].tolist()
        group['exact_force_on_fluid'] = group['exact_force_on_fluid'].tolist()
        if case is None:
            for key in ('exact_flux', 'exact_force_on_fluid', 'traction_squared_error'):
                del group[key]
    return {'by_tag': groups, 'viscous_form': result.viscous_form,
            'load_kind': 'physical_traction' if result.viscous_form == 'symmetric_gradient' else 'laplacian_conormal',
            'net_flux': float(net_flux), 'net_force_on_fluid': net_force.tolist(),
            'stress_force_balance': float(np.linalg.norm(net_force+np.asarray(result.diagnostics['body_force'])
                                                        -np.asarray(result.diagnostics.get('inertial_force', [0., 0.])))),
            'balance_kind': 'unsteady_momentum' if 'inertial_force' in result.diagnostics else 'steady_force',
            'max_tag_force_error': float(force_error) if case is not None else None,
            'traction_l2_error': float(traction_error) if case is not None else None}
