"""Manufactured dimensionless -nu*Delta(u)+grad(p)=f, div(u)=0 cases."""
from dataclasses import dataclass
import numpy as np
from numpy.polynomial import Polynomial


@dataclass(frozen=True)
class StokesCase:
    name: str = 'vortex'
    viscosity: float = 1.0
    pressure_offset: float = 0.0

    def __post_init__(self):
        if self.name not in ('affine', 'channel', 'vortex', 'translated_vortex'):
            raise ValueError('unknown Stokes case')
        if not np.isfinite(self.viscosity) or self.viscosity <= 0:
            raise ValueError('viscosity must be finite and positive')
        if not np.isfinite(self.pressure_offset):
            raise ValueError('pressure offset must be finite')

    def fields(self, points):
        """Return u, grad(u)[component, derivative], p and f; mean(p)=offset."""
        x, y = np.asarray(points).T
        z = np.zeros_like(x)
        if self.name == 'affine':
            u = np.stack((1+x, -.5-y), axis=-1)
            grad = np.zeros((len(x), 2, 2))
            grad[:, 0, 0], grad[:, 1, 1] = 1., -1.
            p, f = x+y-1, np.ones_like(u)
        elif self.name == 'channel':
            u = np.stack((y*(1-y), z), axis=-1)
            grad = np.zeros((len(x), 2, 2))
            grad[:, 0, 1] = 1-2*y
            p, f = self.viscosity*(1-2*x), np.zeros_like(u)
        else:
            a = Polynomial([0, 0, 1, -2, 1])
            ax = [a.deriv(k)(x) for k in range(4)]
            ay = [a.deriv(k)(y) for k in range(4)]
            u = np.stack((ax[0]*ay[1], -ax[1]*ay[0]), axis=-1)
            if self.name == 'translated_vortex':
                u = u + np.array([1., -.5])
            grad = np.stack((ax[1]*ay[1], ax[0]*ay[2],
                             -ax[2]*ay[0], -ax[1]*ay[1]), axis=-1).reshape(-1, 2, 2)
            lap = np.stack((ax[2]*ay[1]+ax[0]*ay[3],
                            -ax[3]*ay[0]-ax[1]*ay[2]), axis=-1)
            p = x*x+y*y-2/3
            f = -self.viscosity*lap + np.stack((2*x, 2*y), axis=-1)
        return u, grad, p+self.pressure_offset, f

    def velocity(self, points):
        return self.fields(points)[0]

    def forcing(self, points):
        return self.fields(points)[3]

    def traction(self, points, normals, viscous_form='symmetric_gradient'):
        """Stress times the fluid domain's outward normal; density is one."""
        _, grad, p, _ = self.fields(points)
        if viscous_form == 'symmetric_gradient':
            grad = grad + grad.transpose(0, 2, 1)
        elif viscous_form != 'laplacian':
            raise ValueError('unknown viscous form')
        return self.viscosity*np.einsum('qij,qj->qi', grad, normals)-p[:, None]*normals
