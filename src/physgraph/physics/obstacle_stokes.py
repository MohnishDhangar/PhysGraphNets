"""Finite-domain, no-slip obstacle cases for steady incompressible Stokes."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ObstacleStokesCase:
    name: str = 'annular_cylinder'
    viscosity: float = 1.
    speed: float = .01
    radius: float = .3
    outer_radius: float = 2.
    half_height: float = .75
    half_length: float = 2.
    chord: float = 1.
    thickness: float = .12
    airfoil_angular_bias: float = 1.

    def __post_init__(self):
        if self.name not in ('annular_cylinder', 'channel_cylinder', 'airfoil'):
            raise ValueError('unknown obstacle Stokes case')
        values = [self.viscosity, self.speed, self.radius, self.outer_radius,
                  self.half_height, self.half_length, self.chord, self.thickness, self.airfoil_angular_bias]
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError('obstacle parameters must be positive and finite')
        if (self.outer_radius <= max(self.radius, self.chord/2)
                or self.radius >= min(self.half_height, self.half_length)
                or not .06 <= self.thickness <= .24 or self.airfoil_angular_bias > 1):
            raise ValueError('invalid obstacle clearance or thickness')

    @property
    def obstacle_tags(self):
        return {'annular_cylinder': ('inner',), 'channel_cylinder': ('cylinder',),
                'airfoil': ('airfoil_upper', 'airfoil_lower')}[self.name]

    @property
    def dirichlet_tags(self):
        if self.name == 'channel_cylinder':
            return ('left', 'top', 'bottom', 'cylinder')
        return None

    @staticmethod
    def zero(points):
        return np.zeros_like(points, dtype=float)

    def uniform(self, points):
        return np.tile([self.speed, 0.], (len(points), 1))

    def inlet(self, points):
        y = np.asarray(points)[:, 1]
        return np.column_stack((self.speed*(1-(y/self.half_height)**2), np.zeros_like(y)))

    @property
    def boundary_velocity(self):
        data = {tag: self.zero for tag in self.obstacle_tags}
        if self.name == 'channel_cylinder':
            data.update(left=self.inlet, top=self.zero, bottom=self.zero)
        else:
            data['outer' if self.name == 'annular_cylinder' else 'farfield'] = self.uniform
        return data

    @property
    def tractions(self):
        return {'right': lambda points, normals: self.zero(points)} if self.name == 'channel_cylinder' else None

    @property
    def coefficients(self):
        """psi=F(r) sin(theta), F=A r+B r^3+C/r+D r log(r)."""
        a, r = self.radius, self.outer_radius
        matrix = np.array([[a, a**3, 1/a, a*np.log(a)],
                           [1, 3*a*a, -1/a**2, np.log(a)+1],
                           [r, r**3, 1/r, r*np.log(r)],
                           [1, 3*r*r, -1/r**2, np.log(r)+1]])
        return np.linalg.solve(matrix, [0., 0., self.speed*r, self.speed])

    def fields(self, points):
        """Analytical circular-annulus solution, smoothly extended over chord slivers.

        This reference is only defined for annular_cylinder. The numerical mesh
        has straight facets: errors include geometry and boundary approximation.
        """
        if self.name != 'annular_cylinder':
            raise ValueError('no analytical field for this obstacle case')
        x, y = np.asarray(points).T
        r = np.sqrt(x*x+y*y)
        if np.any(r <= 0):
            raise ValueError('annular reference excludes the origin')
        a, b, c, d = self.coefficients
        # psi = y H(r); u=(H+y^2 H'/r, -xy H'/r).
        h = a+b*r*r+c/r**2+d*np.log(r)
        k = 2*b-2*c/r**4+d/r**2
        kp = 8*c/r**5-2*d/r**3
        u = np.column_stack((h+y*y*k, -x*y*k))
        du = np.empty((len(x), 2, 2))
        du[:, 0, 0] = x*k+x*y*y*kp/r
        du[:, 0, 1] = 3*y*k+y**3*kp/r
        du[:, 1, 0] = -y*k-x*x*y*kp/r
        du[:, 1, 1] = -x*k-x*y*y*kp/r
        p = self.viscosity*(8*b*x-2*d*x/r**2)
        return u, du, p, np.zeros_like(u)

    @property
    def analytical_drag(self):
        if self.name != 'annular_cylinder':
            raise ValueError('analytical drag is only available for the annular cylinder')
        return float(4*np.pi*self.viscosity*self.coefficients[3])
