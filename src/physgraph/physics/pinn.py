"""Strong-form square cases and boundary liftings for the first PINN baseline."""
from dataclasses import dataclass
from typing import Callable
from physgraph.core.domains import SquareDomain, AnnulusDomain
from physgraph.models.constraints import AnnulusConstraint
from physgraph.physics.annulus import radial_log_case
import numpy as np
from physgraph.physics.base import EllipticProblem, AnalyticalReference
from physgraph.physics.boundary import BoundaryConditions
from physgraph.physics.diffusion import get_elliptic_case
from physgraph.physics.manufactured import get_case

from physgraph.physics.mixed_square import mixed_case, CASE_NAMES as MIXED_CASE_NAMES

DIRICHLET_CASE_NAMES = ('sine', 'affine', 'variable_sine', 'affine_diffusion')
CASE_NAMES = DIRICHLET_CASE_NAMES + MIXED_CASE_NAMES + ('annulus_log',)


@dataclass(frozen=True)
class PINNCase:
    problem: EllipticProblem
    reference: AnalyticalReference
    lifting: tuple[float, float, float]
    diffusion_gradient: tuple[float, float] | Callable[[np.ndarray], np.ndarray]
    boundary_kind: str = 'dirichlet'
    domain: SquareDomain | AnnulusDomain = SquareDomain()
    constraint: AnnulusConstraint | None = None


def get_pinn_case(name: str) -> PINNCase:
    if name not in CASE_NAMES:
        raise ValueError(f'unsupported PINN case {name}; choose from {CASE_NAMES}')
    if name == 'annulus_log':
        domain = AnnulusDomain()
        source = radial_log_case()
        return PINNCase(source.problem, source.reference, (0., 0., 0.), (0., 0.),
                        domain=domain, constraint=AnnulusConstraint(domain))
    if name in MIXED_CASE_NAMES:
        source = mixed_case(name)
        return PINNCase(source.problem, source.reference, (1., 1., 2.), (1., 1.),
                        'right_neumann' if name.endswith('right') else 'left_neumann')
    if name in ('sine', 'affine'):
        source = get_case(name)
        problem = EllipticProblem(name, lambda p: np.ones(len(p)), lambda p: np.zeros(len(p)),
                                  source.forcing, BoundaryConditions(source.exact))
        reference = AnalyticalReference(source.exact, source.gradient)
        lifting = (0., 0., 0.) if name == 'sine' else (1., 2., -3.)
        return PINNCase(problem, reference, lifting, (0., 0.))
    source = get_elliptic_case(name)
    lifting = (1., 1., 2.) if name == 'variable_sine' else (1., 2., -3.)
    return PINNCase(source.problem, source.reference, lifting, (1., 1.))


def diffusion_gradient_at(case, points):
    """Support constant or spatially varying grad(k), with one 2-vector per point."""
    value = case.diffusion_gradient(points) if callable(case.diffusion_gradient) else case.diffusion_gradient
    value = np.asarray(value, dtype=np.float64)
    if value.shape == (2,):
        value = np.broadcast_to(value, (len(points), 2))
    if value.shape != (len(points), 2) or not np.isfinite(value).all():
        raise ValueError('diffusion_gradient must provide a finite 2-vector per point')
    return value
