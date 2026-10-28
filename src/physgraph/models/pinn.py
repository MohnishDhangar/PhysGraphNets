"""Small tanh MLP with exact first/diagonal-second spatial chain rules.

Complex parameter dtypes are preserved for complex-step parameter gradients.
No finite differences in space or analytical solution values enter the model.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class TanhMLP:
    width: int = 8
    depth: int = 2

    def __post_init__(self):
        for name, value in (('width', self.width), ('depth', self.depth)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')

    @property
    def sizes(self):
        return (2,)+(self.width,)*self.depth+(1,)

    @property
    def parameter_count(self):
        return sum(a*b+b for a, b in zip(self.sizes, self.sizes[1:]))

    def initialize(self, rng):
        blocks = []
        for a, b in zip(self.sizes, self.sizes[1:]):
            blocks.extend((rng.normal(0., np.sqrt(2/(a+b)), (a, b)).ravel(), np.zeros(b)))
        return np.concatenate(blocks)

    def derivatives(self, parameters, points):
        """Return N, grad N and diag Hess N, with shapes (P,), (P,2), (P,2)."""
        parameters, points = np.asarray(parameters), np.asarray(points)
        if parameters.shape != (self.parameter_count,) or not np.isfinite(parameters).all():
            raise ValueError('invalid or nonfinite model parameters')
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError('points must have finite shape (P,2)')
        h = 2*points-1
        dh = np.broadcast_to(2*np.eye(2), (len(points), 2, 2)).copy()
        d2h = np.zeros_like(dh)
        offset = 0
        for layer, (a, b) in enumerate(zip(self.sizes, self.sizes[1:])):
            W = parameters[offset:offset+a*b].reshape(a, b)
            offset += a*b
            bias = parameters[offset:offset+b]
            offset += b
            z = h @ W+bias
            dz = np.einsum('nik,ij->njk', dh, W)
            d2z = np.einsum('nik,ij->njk', d2h, W)
            if layer == self.depth:
                h, dh, d2h = z, dz, d2z
            else:
                h = np.tanh(z)
                first = 1-h*h
                second = -2*h*first
                dh = first[:, :, None]*dz
                d2h = second[:, :, None]*dz*dz+first[:, :, None]*d2z
        return h[:, 0], dh[:, 0, :], d2h[:, 0, :]


def constrained_field(model, parameters, points, lifting, boundary_kind='dirichlet', constraint=None):
    """u=b+16*x*(1-x)*y*(1-y)*N, exactly imposing square Dirichlet values.

    lifting=(constant, x coefficient, y coefficient); only boundary information.
    """
    points = np.asarray(points)
    if constraint is not None:
        value, gradient, diagonal = model.derivatives(parameters, constraint.coordinates(points))
        gradient = gradient*constraint.coordinate_scale
        diagonal = diagonal*constraint.coordinate_scale**2
        lift, grad_lift, lap_lift, mask, grad_mask, lap_mask = constraint.numpy_factors(points)
        return (lift+mask*value,
                grad_lift+grad_mask*value[:, None]+mask[:, None]*gradient,
                lap_lift+lap_mask*value+2*np.sum(grad_mask*gradient, axis=1)+mask*np.sum(diagonal, axis=1))
    x, y = points.T
    value, gradient, diagonal = model.derivatives(parameters, points)
    mx, my = x*(1-x), y*(1-y)
    mask = 16*mx*my
    mask_gradient = 16*np.column_stack(((1-2*x)*my, (1-2*y)*mx))
    mask_laplacian = -32*(mx+my)
    if boundary_kind in ('right_neumann', 'left_neumann'):
        z, sign = (x, 1.) if boundary_kind == 'right_neumann' else (1-x, -1.)
        mask = 4*z*my
        mask_gradient = 4*np.column_stack((sign*my, z*(1-2*y)))
        mask_laplacian = -8*z
    elif boundary_kind != 'dirichlet':
        raise ValueError('unsupported boundary kind')
    b0, bx, by = lifting
    result = b0+bx*x+by*y+mask*value
    grad = np.array([bx, by])+mask_gradient*value[:, None]+mask[:, None]*gradient
    laplacian = mask_laplacian*value+2*np.sum(mask_gradient*gradient, axis=1)+mask*np.sum(diagonal, axis=1)
    return result, grad, laplacian
