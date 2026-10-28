"""Autodifferentiable strong-form loss including the variable-diffusion term."""
import torch

from physgraph.models.torch_pinn import field_derivatives, constrained_value


def strong_residual(model, data, *, create_graph=True):
    value, gradient, laplacian = field_derivatives(model, data.points, data.lifting,
                                                  create_graph=create_graph, boundary_kind=data.boundary_kind, constraint=data.constraint)
    return (-data.diffusion*laplacian-torch.sum(gradient*data.diffusion_gradient, dim=1)
            +data.reaction*value-data.forcing)


def loss(model, data, *, create_graph=True):
    residual = strong_residual(model, data, create_graph=create_graph)
    objective = torch.mean(residual.square())/data.loss_scale
    if data.neumann is not None:
        edge, normals, weights, diffusion, target = data.neumann
        with torch.enable_grad():
            coordinates = edge.detach().clone().requires_grad_(True)
            value = constrained_value(model, coordinates, data.lifting, data.boundary_kind, data.constraint)
            gradient = torch.autograd.grad(value.sum(), coordinates, create_graph=create_graph)[0]
            error = diffusion*torch.sum(gradient*normals, dim=1)-target
            objective = objective+data.neumann_weight*torch.sum(weights*error.square())/torch.clamp(torch.sum(weights*target.square()), min=1.)
    return objective
