"""Continuous neural errors using PyTorch's own automatic spatial derivatives."""
import torch

from physgraph.core.torch_runtime import cpu_runtime
from physgraph.evaluation.pinn import evaluate_field
from physgraph.losses.torch_pinn import strong_residual
from physgraph.models.torch_pinn import TorchTanhMLP, constrained_value
from physgraph.physics.torch_pinn import sample_physics


def evaluate_pinn(mesh, model, parameters, case, *, quadrature_order=5,
                  validation_seed=10007, threads=1, holdout_points=1024, boundary_points=257):
    neural = TorchTanhMLP(model.width, model.depth, parameters)

    def value_and_gradient(points):
        coordinates = torch.tensor(points, dtype=torch.float64, device='cpu')
        with torch.enable_grad():
            coordinates.requires_grad_(True)
            value = constrained_value(neural, coordinates, case.lifting, case.boundary_kind, case.constraint)
            gradient = torch.autograd.grad(value.sum(), coordinates)[0]
        return value.detach().numpy(), gradient.detach().numpy()

    def residual_at(points):
        return strong_residual(neural, sample_physics(points, case), create_graph=False).detach().numpy()

    with cpu_runtime(threads):
        return evaluate_field(mesh, case, value_and_gradient, residual_at,
            quadrature_order=quadrature_order, validation_seed=validation_seed,
            holdout_points=holdout_points, boundary_points=boundary_points)
