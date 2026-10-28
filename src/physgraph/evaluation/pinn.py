"""Evaluate the continuous neural field, never substituting it for FEM truth."""
import numpy as np
from physgraph.fem.quadrature import triangle_rule
from physgraph.models.pinn import constrained_field
from physgraph.losses.pinn import strong_residual


def evaluate_pinn(mesh, model, parameters, case, *, quadrature_order=5, validation_seed=10007, holdout_points=1024, boundary_points=257):
    return evaluate_field(mesh, case,
        lambda points: constrained_field(model, parameters, points, case.lifting, case.boundary_kind, case.constraint)[:2],
        lambda points: strong_residual(model, parameters, points, case),
        quadrature_order=quadrature_order, validation_seed=validation_seed,
        holdout_points=holdout_points, boundary_points=boundary_points)


def evaluate_field(mesh, case, value_and_gradient, residual_at, *, quadrature_order=5,
                   validation_seed=10007, holdout_points=1024, boundary_points=257):
    """Shared quadrature/metrics; each backend supplies its own derivatives."""
    validate_evaluation(quadrature_order, validation_seed, holdout_points, boundary_points)
    bary, weights = triangle_rule(quadrature_order)
    vertices = mesh.nodes[mesh.triangles]
    a, b = vertices[:, 1]-vertices[:, 0], vertices[:, 2]-vertices[:, 0]
    areas = .5*(a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0])
    if np.any(areas <= 0):
        raise ValueError('evaluation requires positive triangle areas')
    l2 = h1 = norm_l2 = norm_h1 = 0.
    # Bound quadrature working memory independently of mesh size.
    cells_per_batch = max(1, 4096//len(weights))
    for start in range(0, len(vertices), cells_per_batch):
        stop = start+cells_per_batch
        points = np.einsum('qi,tij->tqj', bary, vertices[start:stop]).reshape(-1, 2)
        integration_weights = (areas[start:stop, None]*weights[None, :]).ravel()
        values, gradients = value_and_gradient(points)
        exact, exact_gradient = case.reference.exact(points), case.reference.gradient(points)
        l2 += float(integration_weights @ (values-exact)**2)
        h1 += float(integration_weights @ np.sum((gradients-exact_gradient)**2, axis=1))
        norm_l2 += float(integration_weights @ exact**2)
        norm_h1 += float(integration_weights @ np.sum(exact_gradient**2, axis=1))
    polygon_integrals = (l2, h1, norm_l2, norm_h1)
    ideal_rule = case.domain.quadrature(quadrature_order)
    if ideal_rule is not None:
        ideal_points, ideal_weights = ideal_rule
        l2 = h1 = norm_l2 = norm_h1 = 0.
        for start in range(0, len(ideal_points), 4096):
            p = ideal_points[start:start+4096]
            w = ideal_weights[start:start+4096]
            u, grad = value_and_gradient(p)
            exact, exact_grad = case.reference.exact(p), case.reference.gradient(p)
            l2 += float(w @ (u-exact)**2)
            h1 += float(w @ np.sum((grad-exact_grad)**2, axis=1))
            norm_l2 += float(w @ exact**2)
            norm_h1 += float(w @ np.sum(exact_grad**2, axis=1))
    prediction = value_and_gradient(mesh.nodes)[0]
    # A separate random stream measures strong-form residual off training points.
    holdout = case.domain.sample(np.random.default_rng(validation_seed), holdout_points)
    residual = residual_at(holdout)
    boundary = case.domain.boundary_points(boundary_points)
    selector = case.problem.boundary.dirichlet_selector
    if selector is not None:
        boundary = boundary[selector(boundary)]
    boundary_values = value_and_gradient(boundary)[0]
    metrics = {
        'relative_l2': float(np.sqrt(l2/max(norm_l2, 1e-24))),
        'relative_h1_seminorm': float(np.sqrt(h1/max(norm_h1, 1e-24))),
        'absolute_l2': float(np.sqrt(l2)), 'absolute_h1_seminorm': float(np.sqrt(h1)),
        'max_nodal_error': float(np.max(np.abs(prediction-case.reference.exact(mesh.nodes)))),
        'max_boundary_error': float(np.max(np.abs(boundary_values-case.problem.boundary.dirichlet(boundary)))),
        'holdout_residual_rms': float(np.sqrt(np.mean(residual**2))),
        'holdout_residual_relative_rms': float(np.sqrt(np.mean(residual**2)/max(float(np.mean(case.problem.forcing(holdout)**2)), 1.))),
        'holdout_points': len(holdout), 'validation_seed': validation_seed,
    }
    if ideal_rule is not None:
        p_l2, p_h1, p_norm_l2, p_norm_h1 = polygon_integrals
        ideal_area = float(np.sum(ideal_weights))
        polygon_area = float(np.sum(areas))
        metrics.update(polygon_relative_l2=float(np.sqrt(p_l2/max(p_norm_l2, 1e-24))),
                       polygon_relative_h1_seminorm=float(np.sqrt(p_h1/max(p_norm_h1, 1e-24))),
                       polygon_absolute_l2=float(np.sqrt(p_l2)), polygon_absolute_h1_seminorm=float(np.sqrt(p_h1)),
                       ideal_domain_area=ideal_area, polygon_domain_area=polygon_area,
                       geometry_area_relative_error=abs(polygon_area-ideal_area)/ideal_area)
    if case.boundary_kind != 'dirichlet':
        # Dense midpoint sample differs from the training Gauss nodes.
        edge = np.column_stack((np.full(boundary_points, float(case.boundary_kind == 'right_neumann')),
                                (np.arange(boundary_points)+.5)/boundary_points))
        normals = np.tile([1. if case.boundary_kind == 'right_neumann' else -1., 0.], (boundary_points, 1))
        gradient = value_and_gradient(edge)[1]
        target = case.problem.boundary.neumann(edge, normals)
        error = case.problem.diffusion(edge)*np.sum(gradient*normals, axis=1)-target
        metrics.update(neumann_flux_rms=float(np.sqrt(np.mean(error**2))),
                       max_neumann_flux_error=float(np.max(np.abs(error))),
                       neumann_flux_relative_rms=float(np.sqrt(np.mean(error**2)/max(float(np.mean(target**2)), 1.))))
    if not all(np.isfinite(value) for value in metrics.values()):
        raise FloatingPointError('nonfinite PINN evaluation')
    return prediction, metrics


def validate_evaluation(quadrature_order, validation_seed, holdout_points, boundary_points):
    for name, value, minimum in (('validation_seed', validation_seed, 0),
            ('holdout_points', holdout_points, 1), ('boundary_points', boundary_points, 2)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    triangle_rule(quadrature_order)
