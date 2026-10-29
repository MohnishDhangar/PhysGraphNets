import numpy as np
import pytest
from physgraph.fluids.periodic import PeriodicFlow
from physgraph.fluids.particles import sample_periodic
from physgraph.physics.periodic_flow import velocity, vorticity, shear_trajectory


@pytest.mark.parametrize('case',['taylor_green','decaying_shear'])
def test_manufactured_velocity_sign_and_incompressibility(case):
    flow=PeriodicFlow(24,.05)
    omega=flow.project(vorticity(flow.grid,0,.05,case))
    np.testing.assert_allclose(flow.velocity(omega),velocity(flow.grid,0,.05,case),atol=1e-14)
    assert flow.diagnostics(omega)['max_divergence']<1e-13


def test_nonlinear_advection_and_viscous_rhs_against_analytic_product_rule():
    flow=PeriodicFlow(24,.07)
    x,y=flow.grid[...,0],flow.grid[...,1]
    omega=2*np.sin(x)*np.sin(y)+np.cos(2*x)*np.sin(y)
    u=np.sin(x)*np.cos(y)+.2*np.cos(2*x)*np.cos(y)
    v=-np.cos(x)*np.sin(y)+.4*np.sin(2*x)*np.sin(y)
    wx=2*np.cos(x)*np.sin(y)-2*np.sin(2*x)*np.sin(y)
    wy=2*np.sin(x)*np.cos(y)+np.cos(2*x)*np.cos(y)
    lap=-4*np.sin(x)*np.sin(y)-5*np.cos(2*x)*np.sin(y)
    rhs,_=flow.rhs(flow.project(omega))
    np.testing.assert_allclose(np.fft.ifft2(rhs).real,-u*wx-v*wy+.07*lap,atol=5e-14)


def test_temporal_fourth_order_convergence_energy_decay_and_final_frame():
    errors=[]
    flow=PeriodicFlow(8,.2)
    initial=vorticity(flow.grid,0,.2)
    for steps in (20,40,80):
        result=flow.integrate(initial,[[.3,.7]],dt=1/steps,steps=steps,save_every=7)
        exact=velocity(flow.grid,1.,.2)
        errors.append(np.linalg.norm(result['velocity'][-1]-exact))
        energy=[r['kinetic_energy_mean'] for r in result['diagnostics']]
        assert np.all(np.diff(energy)<0)
        assert result['times'][-1]==1.
    assert all(np.log2(a/b)>3.8 for a,b in zip(errors,errors[1:]))


def test_particles_converge_spatially_to_exact_shear_trajectories():
    points=np.random.default_rng(21).uniform(-2,8,(40,2))
    errors=[]
    for n in (16,32,64):
        flow=PeriodicFlow(n,.05)
        result=flow.integrate(vorticity(flow.grid,0,.05,'decaying_shear'),points,dt=.005,steps=100,save_every=100)
        exact=shear_trajectory(points,.5,.05)
        errors.append(np.sqrt(np.mean((result['particles_unwrapped'][-1]-exact)**2)))
        assert np.all((result['particles']>=0)&(result['particles']<2*np.pi))
        np.testing.assert_allclose(result['particles_unwrapped'][-1,:,1],points[:,1],atol=1e-14)
    assert all(a/b>3. for a,b in zip(errors,errors[1:]))


def test_particle_interpolation_periodic_seam_and_constant_flow():
    field=np.broadcast_to([2.,-1.],(8,8,2))
    points=np.array([[0,0],[2*np.pi,-2*np.pi],[2*np.pi-.001,.1]])
    np.testing.assert_allclose(sample_periodic(field,points),np.tile([2.,-1.],(3,1)))


def test_reject_invalid_mean_time_step_and_grid():
    with pytest.raises(ValueError):
        PeriodicFlow(9)
    flow=PeriodicFlow(16)
    with pytest.raises(ValueError,match='zero mean'):
        flow.project(np.ones((16,16)))
    with pytest.raises(ValueError,match='time step'):
        flow.integrate(vorticity(flow.grid,0,.05),[[1,1]],dt=1.,steps=1)


def test_flow_experiment_writes_numerical_trajectory_and_manifest(tmp_path):
    from physgraph.experiments.periodic_flow import run_flow
    result=run_flow(output=tmp_path/'flow',resolution=8,steps=3,save_every=2,particles=3,animation=False)
    assert result['metrics']['status']=='passed'
    assert result['source_sha256'] and result['python']
    with np.load(tmp_path/'flow/trajectory.npz',allow_pickle=False) as archive:
        assert archive['velocity'].shape==(3,8,8,2)
        assert archive['particles'].shape==(3,3,2)
        np.testing.assert_allclose(archive['times'],[0,.02,.03])
