"""Manufactured mixed square cases; outward derivative q=k grad(u).n."""
import numpy as np
from physgraph.physics.base import AnalyticalReference, EllipticProblem
from physgraph.physics.boundary import BoundaryConditions
from physgraph.physics.diffusion import ManufacturedEllipticCase
from physgraph.fem.quadrature import interval_rule

CASE_NAMES = ('mixed_sine_right', 'mixed_sine_left')


def mixed_case(name):
    if name not in CASE_NAMES:
        raise ValueError('unsupported mixed square case')
    right = name.endswith('right')
    sign = 1. if right else -1.
    def wave(p):
        x, y = p.T
        z = x if right else 1-x
        return np.sin(np.pi*z/2)*np.sin(np.pi*y)
    def exact(p):
        return 1+p[:, 0]+2*p[:, 1]+wave(p)
    def gradient(p):
        x, y = p.T
        z = x if right else 1-x
        return np.column_stack((1+sign*np.pi/2*np.cos(np.pi*z/2)*np.sin(np.pi*y),
                                2+np.pi*np.sin(np.pi*z/2)*np.cos(np.pi*y)))
    def diffusion(p):
        return 1+p[:, 0]+p[:, 1]
    def reaction(p):
        return np.ones(len(p))
    def forcing(p):
        return 1.25*np.pi**2*diffusion(p)*wave(p)-gradient(p).sum(axis=1)+exact(p)
    def selector(p):
        return np.isclose(p[:, 0], 0 if right else 1) | np.isclose(p[:, 1], 0) | np.isclose(p[:, 1], 1)
    def neumann(p, normals):
        # On the Neumann edge the wave has zero x derivative; only lifting contributes.
        return diffusion(p)*(normals[:, 0]+2*normals[:, 1])
    return ManufacturedEllipticCase(EllipticProblem(name, diffusion, reaction, forcing,
        BoundaryConditions(exact, selector, neumann)), AnalyticalReference(exact, gradient))


def neumann_rule(kind, order=32):
    """Deterministic interior edge quadrature; normals reverse on the left edge."""
    t, weights = interval_rule(order)
    if kind not in ('right_neumann', 'left_neumann'):
        raise ValueError('unsupported Neumann edge')
    right = kind == 'right_neumann'
    points = np.column_stack((np.full(len(t), float(right)), t))
    normals = np.tile([1. if right else -1., 0.], (len(t), 1))
    return points, normals, weights
