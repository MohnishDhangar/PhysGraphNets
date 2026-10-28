"""CPU domain sampling, membership and geometry for neural PDE experiments."""
from dataclasses import dataclass
import numpy as np
from numpy.polynomial.legendre import leggauss


def valid_points(points):
    return points.ndim == 2 and points.shape[1] == 2 and len(points) > 0 and np.isfinite(points).all()


@dataclass(frozen=True)
class SquareDomain:
    name: str = 'unit_square'

    def sample(self, rng, count):
        return np.clip(rng.random((count, 2)), 1e-12, 1-1e-12)

    def contains(self, points, *, strict=False):
        if not valid_points(points):
            return False
        return bool(np.all((points > 0) & (points < 1)) if strict else np.all((points >= 0) & (points <= 1)))

    def boundary_points(self, count):
        t = np.linspace(0, 1, count)
        return np.vstack((np.column_stack((t, t*0)), np.column_stack((t, t*0+1)),
                          np.column_stack((t*0, t)), np.column_stack((t*0+1, t))))

    def mesh(self, resolution):
        from physgraph.mesh.structured import unit_square_mesh
        return unit_square_mesh(resolution)

    def quadrature(self, order):
        return None  # Mesh triangles exactly cover the square.


@dataclass(frozen=True)
class AnnulusDomain:
    inner_radius: float = .5
    outer_radius: float = 1.
    center: tuple[float, float] = (0., 0.)
    name: str = 'annulus'

    def __post_init__(self):
        if (not np.isfinite([self.inner_radius, self.outer_radius]).all()
                or not 0 < self.inner_radius < self.outer_radius):
            raise ValueError('require finite 0 < inner_radius < outer_radius')
        if np.shape(self.center) != (2,) or not np.isfinite(self.center).all():
            raise ValueError('center must contain two finite coordinates')

    def sample(self, rng, count):
        # r^2 uniform, theta uniform => uniform area, not uniform radius.
        u = np.clip(rng.random((count, 2)), 1e-12, 1-1e-12)
        radius = np.sqrt(self.inner_radius**2+(self.outer_radius**2-self.inner_radius**2)*u[:, 0])
        theta = 2*np.pi*u[:, 1]
        return np.column_stack((radius*np.cos(theta), radius*np.sin(theta)))+self.center

    def contains(self, points, *, strict=False):
        if not valid_points(points):
            return False
        r2 = np.sum((points-self.center)**2, axis=1)
        a2, b2 = self.inner_radius**2, self.outer_radius**2
        if strict:
            return bool(np.all((r2 > a2) & (r2 < b2)))
        tolerance = 32*np.finfo(float).eps*b2
        return bool(np.all((r2 >= a2-tolerance) & (r2 <= b2+tolerance)))

    def boundary_points(self, count):
        theta = 2*np.pi*np.arange(count)/count
        directions = np.column_stack((np.cos(theta), np.sin(theta)))
        return np.vstack((self.inner_radius*directions, self.outer_radius*directions))+self.center

    def mesh(self, resolution):
        from physgraph.mesh.annulus import annulus_mesh
        return annulus_mesh(resolution, 8*resolution, self.inner_radius, self.outer_radius, self.center)

    def quadrature(self, order):
        """True circular domain: Gauss in radius, periodic trapezoidal angle."""
        if isinstance(order, bool) or not isinstance(order, int) or order < 1:
            raise ValueError('quadrature order must be a positive integer')
        x, w = leggauss(8*order)
        a, b = self.inner_radius, self.outer_radius
        r = a+(b-a)*(x+1)/2
        theta = 2*np.pi*np.arange(64*order)/(64*order)
        radius, angle = np.meshgrid(r, theta, indexing='ij')
        points = np.column_stack((radius.ravel()*np.cos(angle.ravel()), radius.ravel()*np.sin(angle.ravel())))+self.center
        weights = np.broadcast_to((w*(b-a)/2*r)[:, None]*(2*np.pi/len(theta)), radius.shape).ravel()
        return points, weights
