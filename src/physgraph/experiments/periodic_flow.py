"""Verified periodic flow and particle trajectories for future viewer playback."""
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from physgraph.fluids.periodic import PeriodicFlow
from physgraph.physics.periodic_flow import velocity, vorticity, shear_trajectory
from physgraph.visualization.flow import animate_flow
from physgraph.io.manifest import write_manifest


def run_flow(*, output, case='taylor_green', resolution=32, viscosity=.05, dt=.01,
             steps=200, save_every=5, particles=100, seed=7, animation=True):
    if type(particles) is not int or particles<1 or type(seed) is not int or seed<0:
        raise ValueError('require positive particle count and nonnegative seed')
    flow=PeriodicFlow(resolution,viscosity)
    initial=vorticity(flow.grid,0.,viscosity,case)
    points=np.random.default_rng(seed).uniform(0,2*np.pi,(particles,2))
    dest=Path(output)
    if dest.exists() and (not dest.is_dir() or any(dest.iterdir())):
        raise ValueError('flow output must be empty')
    dest.mkdir(parents=True,exist_ok=True)
    config=dict(case=case,resolution=resolution,viscosity=viscosity,dt=dt,steps=steps,
                save_every=save_every,particles=particles,seed=seed,animation=animation,
                domain='periodic [0,2*pi)^2',method='dealiased Fourier + coupled RK4',
                particle_interpolation='periodic bilinear',velocity_mean=[0.,0.],
                acceptance={'relative_velocity_l2':1e-6,'max_divergence':1e-10},
                fields='numerical reference solver; no GNN predictions',units='dimensionless')
    started=perf_counter()
    write_manifest(dest/'manifest.json',config,{'status':'running'},0.)
    try:
        result=flow.integrate(initial,points,dt=dt,steps=steps,save_every=save_every)
        errors=[]
        for t,field,omega in zip(result['times'],result['velocity'],result['vorticity']):
            exact=velocity(flow.grid,t,viscosity,case)
            exact_omega=vorticity(flow.grid,t,viscosity,case)
            errors.append({'time':float(t),'relative_velocity_l2':float(np.linalg.norm(field-exact)/np.linalg.norm(exact)),
                           'relative_vorticity_l2':float(np.linalg.norm(omega-exact_omega)/np.linalg.norm(exact_omega))})
        metrics={'status':'passed','errors':errors,'diagnostics':result['diagnostics'],
                 'max_relative_velocity_l2':max(r['relative_velocity_l2'] for r in errors),
                 'max_divergence':max(r['max_divergence'] for r in result['diagnostics'])}
        if case=='decaying_shear':
            exact_particles=shear_trajectory(points,steps*dt,viscosity)
            metrics['max_particle_position_error']=float(np.max(np.linalg.norm(result['particles_unwrapped'][-1]-exact_particles,axis=1)))
        if metrics['max_relative_velocity_l2']>1e-6 or metrics['max_divergence']>1e-10:
            metrics['status']='failed_acceptance'
        np.savez_compressed(dest/'trajectory.npz',**{k:v for k,v in result.items() if k!='diagnostics'},
            metadata=np.array(json.dumps({'artifact':'periodic_fluid_trajectory','schema_version':1,
                                         'axes':'frames,y,x,component for velocity; frames,particle,component for positions',**config})))
        if animation:
            animate_flow(result,dest/'particles.gif')
    except Exception as exc:
        write_manifest(dest/'manifest.json',config,{'status':'failed','reason':f'{type(exc).__name__}: {exc}'},perf_counter()-started)
        raise
    return write_manifest(dest/'manifest.json',config,metrics,perf_counter()-started)
