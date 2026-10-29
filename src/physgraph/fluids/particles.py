"""Passive tracers: periodic bilinear interpolation and coupled RK4 stages."""
import numpy as np


def sample_periodic(velocity, points):
    field=np.asarray(velocity)
    p=np.asarray(points)
    if (field.ndim!=3 or field.shape[0]!=field.shape[1] or field.shape[2]!=2
            or len(field)<2 or p.ndim!=2 or p.shape[1]!=2
            or not np.isfinite(field).all() or not np.isfinite(p).all()):
        raise ValueError('require a finite square vector grid and (P,2) particle positions')
    n=len(field)
    coordinates=np.mod(p,2*np.pi)*n/(2*np.pi)
    base=np.floor(coordinates).astype(np.int64)
    fraction=coordinates-base
    ix,iy=base[:,0]%n,base[:,1]%n
    fx,fy=fraction[:,0,None],fraction[:,1,None]
    return ((1-fx)*(1-fy)*field[iy,ix]+fx*(1-fy)*field[iy,(ix+1)%n]
            +(1-fx)*fy*field[(iy+1)%n,ix]+fx*fy*field[(iy+1)%n,(ix+1)%n])


def advance_particles(points, stage_velocities, dt):
    if not np.isfinite(dt) or dt<=0 or len(stage_velocities)!=4:
        raise ValueError('require positive dt and four coupled velocity stages')
    p=np.asarray(points)
    k1=sample_periodic(stage_velocities[0],p)
    k2=sample_periodic(stage_velocities[1],p+dt*k1/2)
    k3=sample_periodic(stage_velocities[2],p+dt*k2/2)
    k4=sample_periodic(stage_velocities[3],p+dt*k3)
    return p+dt*(k1+2*k2+2*k3+k4)/6
