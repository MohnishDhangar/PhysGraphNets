"""Dealiased Fourier vorticity Navier-Stokes reference with coupled RK4 tracers.

omega_t + u.grad(omega) = nu*Delta(omega); div(u)=0. Periodic box,
zero mean velocity and vorticity; no obstacles, pressure output or wall model.
"""
import numpy as np
from physgraph.fluids.particles import advance_particles


class PeriodicFlow:
    def __init__(self, resolution=32, viscosity=.05):
        if type(resolution) is not int or resolution<8 or resolution%2:
            raise ValueError('resolution must be an even integer >=8')
        if isinstance(viscosity,bool) or not np.isfinite(viscosity) or viscosity<=0:
            raise ValueError('viscosity must be positive and finite')
        self.n, self.viscosity=resolution,viscosity
        axis=np.arange(resolution)*2*np.pi/resolution
        self.grid=np.stack(np.meshgrid(axis,axis,indexing='xy'),axis=-1)
        frequencies=np.fft.fftfreq(resolution,d=1/resolution)
        self.kx,self.ky=np.meshgrid(frequencies,frequencies,indexing='xy')
        self.k2=self.kx**2+self.ky**2
        self.mask=(np.abs(self.kx)<resolution/3)&(np.abs(self.ky)<resolution/3)
        self.inverse_k2=np.divide(1.,self.k2,out=np.zeros_like(self.k2),where=self.k2>0)

    def project(self, omega):
        omega=np.asarray(omega)
        if omega.shape!=(self.n,self.n) or not np.isrealobj(omega) or not np.isfinite(omega).all():
            raise ValueError('initial vorticity must be a finite real grid')
        if abs(omega.mean())>1e-12*max(1.,float(np.max(np.abs(omega)))):
            raise ValueError('periodic vorticity must have zero mean')
        result=np.fft.fft2(omega)*self.mask
        result[0,0]=0
        return result

    def velocity(self, omega_hat):
        psi=omega_hat*self.inverse_k2
        return np.stack((np.fft.ifft2(1j*self.ky*psi).real,
                         np.fft.ifft2(-1j*self.kx*psi).real),axis=-1)

    def rhs(self, omega_hat):
        flow=self.velocity(omega_hat)
        wx=np.fft.ifft2(1j*self.kx*omega_hat).real
        wy=np.fft.ifft2(1j*self.ky*omega_hat).real
        nonlinear=np.fft.fft2(flow[...,0]*wx+flow[...,1]*wy)*self.mask
        nonlinear[0,0]=0
        return -nonlinear-self.viscosity*self.k2*omega_hat,flow

    def diagnostics(self, omega_hat):
        flow=self.velocity(omega_hat)
        divergence=np.fft.ifft2(1j*self.kx*np.fft.fft2(flow[...,0])+1j*self.ky*np.fft.fft2(flow[...,1])).real
        omega=np.fft.ifft2(omega_hat).real
        return {'kinetic_energy_mean':float(.5*np.mean(np.sum(flow**2,axis=-1))),
                'enstrophy_mean':float(.5*np.mean(omega**2)),
                'max_divergence':float(np.max(np.abs(divergence))),
                'mean_vorticity':float(omega.mean())}

    def integrate(self, initial_vorticity, particles, *, dt=.01, steps=100, save_every=5):
        if (type(steps) is not int or steps<1 or type(save_every) is not int or save_every<1
                or isinstance(dt,bool) or not np.isfinite(dt) or dt<=0):
            raise ValueError('require positive finite dt and positive integer steps/save_every')
        p=np.array(particles,dtype=float,copy=True)
        if p.ndim!=2 or p.shape[1]!=2 or len(p)==0 or not np.isfinite(p).all():
            raise ValueError('require nonempty finite (P,2) particles')
        omega=self.project(initial_vorticity)
        frames=[]; positions=[]; vorticities=[]; times=[]; diagnostics=[]
        def save(step):
            times.append(step*dt)
            frames.append(self.velocity(omega))
            vorticities.append(np.fft.ifft2(omega).real)
            positions.append(p.copy())
            diagnostics.append(self.diagnostics(omega))
        save(0)
        for step in range(1,steps+1):
            a,u1=self.rhs(omega)
            b,u2=self.rhs(omega+dt*a/2)
            c,u3=self.rhs(omega+dt*b/2)
            d,u4=self.rhs(omega+dt*c)
            # Conservative spectral advection/diffusion step restriction.
            for velocity in (u1,u2,u3,u4):
                bound=(np.max(np.abs(velocity[...,0]))+np.max(np.abs(velocity[...,1])))*self.n/3
                bound+=self.viscosity*np.max(self.k2[self.mask])
                if not np.isfinite(bound) or dt*bound>.5:
                    raise ValueError('time step exceeds conservative advection/diffusion limit; reduce dt')
            omega=(omega+dt*(a+2*b+2*c+d)/6)*self.mask
            p=advance_particles(p,(u1,u2,u3,u4),dt)
            if not np.isfinite(omega).all() or not np.isfinite(p).all():
                raise ValueError('nonfinite fluid/particle state')
            if step%save_every==0 or step==steps:
                save(step)
        return {'times':np.array(times), 'velocity':np.array(frames),
                'vorticity':np.array(vorticities), 'particles_unwrapped':np.array(positions),
                'particles':np.mod(positions,2*np.pi), 'grid':self.grid.copy(),
                'diagnostics':diagnostics}
