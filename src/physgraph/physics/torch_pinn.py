"""Sample fixed physical coefficients/forcing without analytical solution data."""
from dataclasses import dataclass

import numpy as np
import torch

from physgraph.physics.base import sample_scalar
from physgraph.physics.pinn import diffusion_gradient_at


@dataclass(frozen=True)
class TensorPDEData:
    points: torch.Tensor
    diffusion: torch.Tensor
    reaction: torch.Tensor
    forcing: torch.Tensor
    diffusion_gradient: torch.Tensor
    lifting: tuple[float, float, float]
    loss_scale: torch.Tensor
    boundary_kind: str
    neumann: tuple | None
    neumann_weight: float
    constraint: object


def sample_physics(points, case, *, neumann_points=32, neumann_weight=1.) -> TensorPDEData:
    points = np.asarray(points, dtype=np.float64)
    if not case.domain.contains(points):
        raise ValueError('PINN points must lie inside the case domain')
    k = sample_scalar(case.problem.diffusion, points, 'diffusion')
    c = sample_scalar(case.problem.reaction, points, 'reaction')
    f = sample_scalar(case.problem.forcing, points, 'forcing')
    grad_k = diffusion_gradient_at(case, points)
    if np.any(k <= 0) or np.any(c < 0):
        raise ValueError('invalid diffusion, reaction or diffusion gradient')
    tensor = lambda value: torch.tensor(value, dtype=torch.float64, device='cpu')
    forcing = tensor(f)
    boundary = None
    if case.boundary_kind != 'dirichlet':
        from physgraph.physics.mixed_square import neumann_rule
        edge, normals, weights = neumann_rule(case.boundary_kind, neumann_points)
        target = sample_scalar(lambda p: case.problem.boundary.neumann(p, normals), edge, 'Neumann data')
        boundary = tuple(tensor(v) for v in (edge, normals, weights,
                         case.problem.diffusion(edge), target))
    return TensorPDEData(tensor(points), tensor(k), tensor(c), forcing, tensor(grad_k),
                         case.lifting, torch.clamp(torch.mean(forcing**2), min=1.), case.boundary_kind, boundary, neumann_weight, case.constraint)
