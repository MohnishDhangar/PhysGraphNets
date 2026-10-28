"""Manufactured variable-coefficient diffusion/reaction and mixed-BC cases."""
from dataclasses import dataclass

import numpy as np

from physgraph.physics.base import AnalyticalReference, EllipticProblem
from physgraph.physics.boundary import BoundaryConditions

CASE_NAMES = ("variable_sine", "affine_diffusion", "mixed_polynomial")


@dataclass(frozen=True)
class ManufacturedEllipticCase:
    problem: EllipticProblem
    reference: AnalyticalReference


def _diffusion(p):
    return 1 + p[:, 0] + p[:, 1]


def get_elliptic_case(name: str) -> ManufacturedEllipticCase:
    """Return independently specified u, grad u and manufactured forcing."""
    if name == "variable_sine":
        def exact(p):
            x, y = p.T
            return 1 + x + 2*y + np.sin(np.pi*x)*np.sin(np.pi*y)

        def gradient(p):
            x, y = p.T
            return np.column_stack((1 + np.pi*np.cos(np.pi*x)*np.sin(np.pi*y),
                                    2 + np.pi*np.sin(np.pi*x)*np.cos(np.pi*y)))

        def reaction(p):
            return 1 + p[:, 0]

        def forcing(p):
            x, y = p.T
            sine = np.sin(np.pi*x)*np.sin(np.pi*y)
            return (2*np.pi**2*(1+x+y)*sine - 3
                    - np.pi*np.cos(np.pi*x)*np.sin(np.pi*y)
                    - np.pi*np.sin(np.pi*x)*np.cos(np.pi*y)
                    + (1+x)*(1+x+2*y+sine))

        boundary = BoundaryConditions(dirichlet=exact)
    elif name == "affine_diffusion":
        def exact(p):
            return 1 + 2*p[:, 0] - 3*p[:, 1]

        def gradient(p):
            return np.tile([2., -3.], (len(p), 1))

        def reaction(p):
            return np.full(len(p), 2.)

        def forcing(p):
            return 3 + 4*p[:, 0] - 6*p[:, 1]

        boundary = BoundaryConditions(dirichlet=exact)
    elif name == "mixed_polynomial":
        def exact(p):
            return 1 + p[:, 0]**2 + p[:, 1]**2

        def gradient(p):
            return 2*p

        def reaction(p):
            return np.ones(len(p))

        def forcing(p):
            x, y = p.T
            return -4*(1+x+y) - 2*x - 2*y + 1+x*x+y*y

        def selector(p):
            return np.isclose(p[:, 0], 0) | np.isclose(p[:, 1], 0) | np.isclose(p[:, 1], 1)

        def neumann(p, normals):
            return (1+p[:, 0]+p[:, 1]) * np.sum(2*p*normals, axis=1)

        boundary = BoundaryConditions(exact, selector, neumann)
    else:
        raise ValueError(f"Unknown elliptic case {name!r}; choose from {CASE_NAMES}")
    return ManufacturedEllipticCase(
        EllipticProblem(name, _diffusion, reaction, forcing, boundary),
        AnalyticalReference(exact, gradient),
    )
