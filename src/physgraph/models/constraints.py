"""Boundary-data-only hard constraints, independent of analytical references."""
from dataclasses import dataclass
import numpy as np
from physgraph.core.domains import AnnulusDomain


@dataclass(frozen=True)
class AnnulusConstraint:
    domain: AnnulusDomain
    inner_value: float = 1.
    outer_value: float = 2.

    def __post_init__(self):
        if not np.isfinite([self.inner_value, self.outer_value]).all():
            raise ValueError('boundary values must be finite')

    @property
    def coordinate_scale(self):
        return 1/(2*self.domain.outer_radius)

    def coordinates(self, points):
        # Scalar operations work with both NumPy and Torch; no tensor detachment.
        result = points*self.coordinate_scale + .5
        result[:, 0] -= self.domain.center[0]*self.coordinate_scale
        result[:, 1] -= self.domain.center[1]*self.coordinate_scale
        return result

    def values(self, points):
        x = points[:, 0]-self.domain.center[0]
        y = points[:, 1]-self.domain.center[1]
        a, b = self.domain.inner_radius, self.domain.outer_radius
        t = (x*x+y*y-a*a)/(b*b-a*a)
        lift = self.inner_value+(self.outer_value-self.inner_value)*t
        return lift, 4*t*(1-t)

    def numpy_factors(self, points):
        delta = points-np.asarray(self.domain.center)
        a, b = self.domain.inner_radius, self.domain.outer_radius
        denominator = b*b-a*a
        t = (np.sum(delta*delta, axis=1)-a*a)/denominator
        grad_t = 2*delta/denominator
        lap_t = 4/denominator
        difference = self.outer_value-self.inner_value
        lift, mask = self.values(points)
        grad_mask = 4*(1-2*t[:, None])*grad_t
        lap_mask = 4*(1-2*t)*lap_t-8*np.sum(grad_t*grad_t, axis=1)
        return lift, difference*grad_t, difference*lap_t, mask, grad_mask, lap_mask
