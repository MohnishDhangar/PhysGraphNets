"""Strong-form residual, normalized least-squares loss and parameter derivatives."""
import numpy as np
from physgraph.models.pinn import constrained_field
from physgraph.physics.pinn import diffusion_gradient_at


def strong_residual(model, parameters, points, case):
    value, gradient, laplacian = constrained_field(model, parameters, points, case.lifting, case.boundary_kind, case.constraint)
    return (-case.problem.diffusion(points)*laplacian
            -np.sum(gradient*diffusion_gradient_at(case, points), axis=1)
            +case.problem.reaction(points)*value-case.problem.forcing(points))


def loss(model, parameters, points, case, *, neumann_points=32, neumann_weight=1.):
    r = strong_residual(model, parameters, points, case)
    scale = max(float(np.mean(case.problem.forcing(points)**2)), 1.)
    # Deliberately r*r (not abs(r)**2): analytic extension for complex-step.
    objective = np.mean(r*r)/scale
    if case.boundary_kind != 'dirichlet':
        from physgraph.physics.mixed_square import neumann_rule
        edge, normals, weights = neumann_rule(case.boundary_kind, neumann_points)
        grad = constrained_field(model, parameters, edge, case.lifting, case.boundary_kind, case.constraint)[1]
        target = case.problem.boundary.neumann(edge, normals)
        flux_error = case.problem.diffusion(edge)*np.sum(grad*normals, axis=1)-target
        objective += neumann_weight*(weights @ (flux_error*flux_error))/max(float(weights @ (target**2)), 1.)
    return objective


def loss_gradient(model, parameters, points, case, **options):
    """Complex-step derivative, one perturbation per real trainable parameter.

    Smooth tanh, polynomial masks and sums/products are complex analytic near
    the real parameters. Avoids subtractive finite-difference cancellation.
    This CPU baseline trades speed for a short independently testable derivative.
    """
    parameters = np.asarray(parameters, dtype=np.float64)
    gradient = np.empty_like(parameters)
    perturbed = parameters.astype(np.complex128)
    step = 1e-30
    for i in range(len(parameters)):
        perturbed[i] += 1j*step
        gradient[i] = np.imag(loss(model, perturbed, points, case, **options))/step
        perturbed[i] = parameters[i]
    if not np.isfinite(gradient).all():
        raise FloatingPointError('nonfinite PINN parameter gradient')
    return gradient
