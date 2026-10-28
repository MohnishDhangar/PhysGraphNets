"""CPU float64 tanh MLP and automatic spatial derivatives of the full field."""
import numpy as np
import torch
from torch import nn

from physgraph.models.pinn import TanhMLP


class TorchTanhMLP(nn.Module):
    """Flat parameters share the reference W[in,out], bias layout for parity.

    Initial parameters are supplied explicitly, avoiding changes to Torch's RNG.
    Each sample is independent: no batch normalization or other batch coupling.
    """

    def __init__(self, width: int, depth: int, parameters):
        super().__init__()
        architecture = TanhMLP(width, depth)
        values = np.asarray(parameters)
        if (values.shape != (architecture.parameter_count,) or not np.isfinite(values).all()
                or np.iscomplexobj(values)):
            raise ValueError('parameters must be a finite real vector matching the architecture')
        self.sizes = architecture.sizes
        self.coefficients = nn.Parameter(torch.tensor(values, dtype=torch.float64, device='cpu'))

    def forward(self, points):
        if (points.ndim != 2 or points.shape[1] != 2 or points.dtype != torch.float64
                or points.device.type != 'cpu' or self.coefficients.device.type != 'cpu'
                or self.coefficients.dtype != torch.float64):
            raise ValueError('the PyTorch baseline requires CPU float64 parameters and (N,2) points')
        h = 2*points-1
        offset = 0
        for layer, (a, b) in enumerate(zip(self.sizes, self.sizes[1:])):
            weight = self.coefficients[offset:offset+a*b].reshape(a, b)
            offset += a*b
            bias = self.coefficients[offset:offset+b]
            offset += b
            h = h @ weight+bias
            if layer < len(self.sizes)-2:
                h = torch.tanh(h)
        return h[:, 0]


def constrained_value(model, points, lifting, boundary_kind='dirichlet', constraint=None):
    if constraint is not None:
        lift, mask = constraint.values(points)
        return lift+mask*model(constraint.coordinates(points))
    x, y = points[:, 0], points[:, 1]
    b0, bx, by = lifting
    if boundary_kind == 'dirichlet':
        mask = 16*x*(1-x)*y*(1-y)
    elif boundary_kind in ('right_neumann', 'left_neumann'):
        z = x if boundary_kind == 'right_neumann' else 1-x
        mask = 4*z*y*(1-y)
    else:
        raise ValueError('unsupported boundary kind')
    return b0+bx*x+by*y+mask*model(points)


def field_derivatives(model, points, lifting, *, create_graph: bool = True, boundary_kind='dirichlet', constraint=None):
    """Differentiate u=b+mask*N including mask derivatives, returning u,grad,Δu.

    The first derivative needs its graph even for evaluation so the second
    derivative exists. Training also retains the second-derivative graph for
    parameter backpropagation. Summing outputs is valid for this pointwise MLP.
    """
    with torch.enable_grad():
        coordinates = points.detach().clone().requires_grad_(True)
        value = constrained_value(model, coordinates, lifting, boundary_kind, constraint)
        gradient = torch.autograd.grad(value.sum(), coordinates, create_graph=True)[0]
        diagonal = [torch.autograd.grad(gradient[:, d].sum(), coordinates,
                    create_graph=create_graph, retain_graph=True)[0][:, d] for d in range(2)]
        laplacian = diagonal[0]+diagonal[1]
    return value, gradient, laplacian
