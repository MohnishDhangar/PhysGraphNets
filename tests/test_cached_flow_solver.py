"""Reuse is preconditioning only: new Jacobians and right-hand sides must win."""
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from physgraph.fem.cached_flow_solver import CachedFlowSolver
from physgraph.fem.resident_gmres import resident_gmres
from physgraph.fem.navier_stokes import solve_navier_stokes
from physgraph.mesh.structured import tagged_unit_square_mesh
from physgraph.physics.navier_stokes import NavierStokesCase


def test_current_matrix_workspace_reset_and_refresh():
    rng=np.random.default_rng(4)
    a=csr_matrix(rng.normal(size=(12,12))+8*np.eye(12))
    solver=CachedFlowSolver('cpu');solver.begin_step(1)
    try:
        for step in range(1,5):
            solver.begin_step(step)
            current=a+csr_matrix(np.diag(np.linspace(0,.3*step,12)))
            b=rng.normal(size=12)
            np.testing.assert_allclose(solver(current,b),np.linalg.solve(current.toarray(),b),rtol=0,atol=1e-10)
        assert solver.setup_count==1
        assert not solver.records[-1]['preconditioner_rebuilt']
        solver.begin_step(33);solver(a,np.ones(12))
        assert solver.setup_count==2
        with pytest.raises(ValueError,match='finite'):
            solver(a,np.full(12,np.nan))
    finally:solver.close()


def test_cached_trajectory_nonzero_boundary_and_restart():
    case=NavierStokesCase('translated_vortex',pressure_offset=2.5)
    mesh=tagged_unit_square_mesh(3)
    def run(backend,steps=5,state=None):
        return solve_navier_stokes(mesh,case.forcing,case.velocity,lambda x:case.velocity(x,0),
            dt=.02,steps=steps,dirichlet_tags={'left','top','bottom'},tractions={'right':case.traction},
            linear_backend=backend,problem_id='cached-vortex:v1',restart_state=state)
    expected=run('scipy');candidate=run('cpu_cached')
    split=run('cpu_cached',steps=2)
    resumed=run('cpu_cached',steps=3,state=split.restart_state)
    for key in ('velocity','pressure','reactions'):
        np.testing.assert_allclose(getattr(candidate,key),getattr(expected,key),atol=1e-9,rtol=0)
        np.testing.assert_allclose(getattr(resumed,key),getattr(candidate,key)[2:],atol=1e-9,rtol=0)
    assert all(d['max_boundary_error']==0 for d in candidate.diagnostics)
