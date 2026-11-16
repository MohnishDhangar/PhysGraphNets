"""Real-device checks, explicitly opt-in; never substitute CPU for GPU execution."""
import os
import numpy as np
import pytest
from physgraph.fem.navier_stokes import solve_navier_stokes
from physgraph.mesh.structured import tagged_unit_square_mesh
from physgraph.physics.navier_stokes import NavierStokesCase
from physgraph.evaluation.stokes import stokes_errors

pytestmark = pytest.mark.skipif(os.environ.get('PHYSGRAPH_GPU_TEST')!='1',
                               reason='explicit real GPU test run required')


@pytest.mark.parametrize('traction',[False,True])
@pytest.mark.parametrize('scheme',['be','bdf2'])
@pytest.mark.parametrize('gpu_backend',['rocm_gmres','rocm_cached'])
def test_gpu_nonzero_boundary_pressure_and_restart(traction,scheme,gpu_backend):
    case=NavierStokesCase('accelerating_shear',time_profile='linear',pressure_offset=2.5 if traction else 0.)
    mesh=tagged_unit_square_mesh(3)
    def run(backend,steps=3,state=None):
        return solve_navier_stokes(mesh,case.forcing,case.velocity,lambda x:case.velocity(x,0),
            dt=.03,steps=steps,scheme=scheme,problem_id='gpu-patch:v1',
            dirichlet_tags={'left','top','bottom'} if traction else None,
            tractions={'right':case.traction} if traction else None,
            restart_state=state,linear_backend=backend)
    cpu,gpu=run('scipy'),run(gpu_backend)
    for key in ('velocity','pressure','reactions'):
        np.testing.assert_allclose(getattr(gpu,key),getattr(cpu,key),atol=1e-9,rtol=0)
    assert any(r.get('device_name') for r in gpu.linear_solver_records)
    assert all(d['max_boundary_error']==0 for d in gpu.diagnostics)
    for step in range(1,4):
        errors=stokes_errors(gpu.frame(step),case.at(gpu.times[step]))
        assert max(errors[k] for k in ('velocity_l2','velocity_h1_seminorm','pressure_l2'))<1e-9
    first=run(gpu_backend,steps=1)
    resumed=run(gpu_backend,steps=2,state=first.restart_state)
    for key in ('velocity','pressure','reactions'):
        np.testing.assert_allclose(getattr(resumed,key),getattr(gpu,key)[1:],atol=1e-9,rtol=0)
    with pytest.raises(ValueError,match='restart contract'):
        run('scipy',steps=2,state=first.restart_state)
