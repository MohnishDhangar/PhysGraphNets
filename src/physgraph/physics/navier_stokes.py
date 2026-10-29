"""Manufactured unsteady incompressible flow with physical Newtonian stress."""
from dataclasses import dataclass
import numpy as np
from physgraph.physics.stokes import StokesCase


@dataclass(frozen=True)
class NavierStokesCase:
    name: str = 'vortex'
    viscosity: float = .1
    time_profile: str = 'exponential'
    pressure_offset: float = 0.

    def __post_init__(self):
        if self.name not in ('accelerating_shear', 'vortex', 'translated_vortex'):
            raise ValueError('unknown transient Navier-Stokes case')
        if self.time_profile not in ('linear', 'exponential'):
            raise ValueError('unknown time profile')
        if not np.isfinite(self.viscosity) or self.viscosity <= 0:
            raise ValueError('viscosity must be finite and positive')
        if not np.isfinite(self.pressure_offset):
            raise ValueError('pressure offset must be finite')

    def amplitude(self, time):
        if not np.isfinite(time) or time < 0:
            raise ValueError('time must be finite and nonnegative')
        a = np.exp(time) if self.time_profile == 'exponential' else 1+time
        return a, a if self.time_profile == 'exponential' else 1.

    def fields(self, points, time):
        """Return u, grad(u)[component, derivative], p and continuous forcing."""
        points = np.asarray(points)
        x, y = points.T
        if self.name == 'accelerating_shear':
            base = np.column_stack((1+y*(1-y), np.full_like(y, .5)))
            grad = np.zeros((len(x), 2, 2))
            grad[:, 0, 1] = 1-2*y
            p = x+y-1
            steady = np.tile([2*self.viscosity+1, 1.], (len(x), 1))
        else:
            base, grad, p, f = StokesCase('vortex', self.viscosity).fields(points)
            base, grad = 32*base, 32*grad
            gp = 2*points
            steady = 32*(f-gp)+gp
            if self.name == 'translated_vortex':
                base += [1., -.5]
        a, derivative = self.amplitude(time)
        advection = np.einsum('qij,qj->qi', grad, base)
        force = derivative*base + a*a*advection + a*steady
        return a*base, a*grad, a*p+self.pressure_offset, force

    def velocity(self, points, time):
        return self.fields(points, time)[0]

    def forcing(self, points, time):
        return self.fields(points, time)[3]

    def traction(self, points, normals, time):
        _, grad, p, _ = self.fields(points, time)
        return self.viscosity*np.einsum('qij,qj->qi', grad+grad.transpose(0, 2, 1), normals)-p[:, None]*normals

    def at(self, time):
        """Adapter for existing physical-space field and traction quadrature."""
        case = self

        class Snapshot:
            def fields(self, points):
                return case.fields(points, time)

            def velocity(self, points):
                return case.velocity(points, time)

            def traction(self, points, normals, viscous_form='symmetric_gradient'):
                if viscous_form != 'symmetric_gradient':
                    raise ValueError('transient reference requires physical stress')
                return case.traction(points, normals, time)

        return Snapshot()
