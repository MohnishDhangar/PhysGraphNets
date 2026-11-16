"""DFG 2D-3 fixed-domain, zero-start cylinder-flow specification."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CylinderFlowCase:
    name: str = 'dfg_2d3'
    viscosity: float = .001
    length: float = 2.2
    height: float = .41
    radius: float = .05
    center: tuple = (.2, .2)
    peak_speed: float = 1.5
    duration: float = 8.

    def __post_init__(self):
        if self.name != 'dfg_2d3':
            raise ValueError('unknown cylinder flow case')
        if (not np.isfinite([self.viscosity, self.length, self.height, self.radius,
                            *self.center, self.peak_speed, self.duration]).all()
                or min(self.viscosity, self.length, self.height, self.radius, self.peak_speed, self.duration) <= 0):
            raise ValueError('cylinder parameters must be positive and finite')
        x, y = self.center
        if min(x, self.length-x, y, self.height-y) <= self.radius:
            raise ValueError('cylinder must lie strictly inside channel')

    @staticmethod
    def zero(points, time=0.):
        return np.zeros_like(points, dtype=float)

    def inlet(self, points, time):
        if not np.isfinite(time) or not 0 <= time <= self.duration+1e-12:
            raise ValueError('cylinder time must lie in the specified interval')
        y = np.asarray(points)[:, 1]
        speed = self.peak_speed*np.sin(np.pi*time/self.duration)
        return np.column_stack((4*speed*y*(self.height-y)/self.height**2, np.zeros_like(y)))

    @property
    def boundary_velocity(self):
        return {'left': self.inlet, 'top': self.zero, 'bottom': self.zero, 'cylinder': self.zero}

    @property
    def dirichlet_tags(self):
        return ('left', 'top', 'bottom', 'cylinder')

    @property
    def tractions(self):
        return {'right': lambda x,n,t: self.zero(x,t)}

    @property
    def force_scale(self):
        # Fixed peak mean inlet velocity, never the instantaneous vanishing speed.
        return .5*(2*self.peak_speed/3)**2*(2*self.radius)

    @property
    def matches_published_case(self):
        return self == CylinderFlowCase()
