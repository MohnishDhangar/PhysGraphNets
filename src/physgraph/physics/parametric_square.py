"""Two-mode manufactured fields for -div(k grad u) + c u = f on [0,1]^2."""
import numpy as np

from physgraph.physics.base import AnalyticalReference, EllipticProblem
from physgraph.physics.boundary import BoundaryConditions
from physgraph.physics.diffusion import ManufacturedEllipticCase


DEFAULTS = dict(offset=1., slope_x=.5, slope_y=-.25, amplitude_1=.7,
                amplitude_2=-.3, diffusion_base=1., diffusion_x=.4,
                diffusion_y=.2, reaction=.5, boundary_mode='dirichlet')
BOUNDARY_MODES = ('dirichlet', 'right_neumann')


def square_case(*, offset=1., slope_x=.5, slope_y=-.25, amplitude_1=.7,
                amplitude_2=-.3, diffusion_base=1., diffusion_x=.4,
                diffusion_y=.2, reaction=.5, boundary_mode='dirichlet'):
    """Affine boundary field plus sin(pi*x)sin(pi*y), sin(2*pi*x)sin(pi*y).

    k = diffusion_base + diffusion_x*x + diffusion_y*y is strictly positive;
    c = reaction >= 0. Mixed cases prescribe q=k grad(u).n on x=1.
    The forcing uses the analytic product rule, independent of FEM assembly.
    """
    parameters = (offset, slope_x, slope_y, amplitude_1, amplitude_2,
                  diffusion_base, diffusion_x, diffusion_y, reaction)
    if (any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float)) for v in parameters)
            or not np.isfinite(parameters).all()):
        raise ValueError('square parameters must be finite real numbers')
    if diffusion_base <= 0 or min(diffusion_x, diffusion_y, reaction) < 0:
        raise ValueError('require positive diffusion_base and nonnegative diffusion slopes/reaction')
    if boundary_mode not in BOUNDARY_MODES:
        raise ValueError(f'boundary_mode must be one of {BOUNDARY_MODES}')

    def exact(p):
        x, y = p.T
        return (offset+slope_x*x+slope_y*y
                + (amplitude_1*np.sin(np.pi*x)+amplitude_2*np.sin(2*np.pi*x))*np.sin(np.pi*y))

    def gradient(p):
        x, y = p.T
        return np.column_stack((
            slope_x+np.pi*(amplitude_1*np.cos(np.pi*x)+2*amplitude_2*np.cos(2*np.pi*x))*np.sin(np.pi*y),
            slope_y+np.pi*(amplitude_1*np.sin(np.pi*x)+amplitude_2*np.sin(2*np.pi*x))*np.cos(np.pi*y)))

    def diffusion(p):
        return diffusion_base+diffusion_x*p[:, 0]+diffusion_y*p[:, 1]

    def forcing(p):
        x, y = p.T
        minus_laplacian = np.pi**2*(2*amplitude_1*np.sin(np.pi*x)+5*amplitude_2*np.sin(2*np.pi*x))*np.sin(np.pi*y)
        return diffusion(p)*minus_laplacian-gradient(p) @ np.array([diffusion_x, diffusion_y])+reaction*exact(p)

    def neumann(p, normals):
        return diffusion(p)*np.sum(gradient(p)*normals, axis=1)

    boundary = BoundaryConditions(exact) if boundary_mode == 'dirichlet' else BoundaryConditions(
        exact, lambda p: ~np.isclose(p[:, 0], 1.), neumann)
    return ManufacturedEllipticCase(
        EllipticProblem(f'parametric_square_{boundary_mode}', diffusion,
                        lambda p: np.full(len(p), reaction), forcing, boundary),
        AnalyticalReference(exact, gradient))
