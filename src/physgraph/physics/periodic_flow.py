"""Exact unforced incompressible solutions on the periodic [0,2*pi)^2 box."""
import numpy as np

CASES = ('taylor_green', 'decaying_shear')


def velocity(points, time, viscosity, case='taylor_green'):
    if case not in CASES or not np.isfinite([time,viscosity]).all() or time < 0 or viscosity <= 0:
        raise ValueError('require supported case, nonnegative time and positive viscosity')
    p=np.asarray(points)
    if p.shape[-1:] != (2,) or not np.isfinite(p).all():
        raise ValueError('points must be finite 2D coordinates')
    x,y=p[...,0],p[...,1]
    if case=='taylor_green':
        scale=np.exp(-2*viscosity*time)
        return scale*np.stack((np.sin(x)*np.cos(y),-np.cos(x)*np.sin(y)),axis=-1)
    return np.exp(-viscosity*time)*np.stack((np.sin(y),np.zeros_like(x)),axis=-1)


def vorticity(points, time, viscosity, case='taylor_green'):
    # omega = dv/dx - du/dy, u = (psi_y, -psi_x), -Delta psi = omega.
    velocity(points,time,viscosity,case)  # Shared parameter/domain validation.
    x,y=np.asarray(points)[...,0],np.asarray(points)[...,1]
    if case=='taylor_green':
        return 2*np.exp(-2*viscosity*time)*np.sin(x)*np.sin(y)
    return -np.exp(-viscosity*time)*np.cos(y)


def shear_trajectory(initial, time, viscosity):
    velocity(initial,time,viscosity,'decaying_shear')
    points=np.array(initial,dtype=float,copy=True)
    points[...,0] += np.sin(points[...,1])*(-np.expm1(-viscosity*time))/viscosity
    return points
