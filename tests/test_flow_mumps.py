"""Optional backend preserves boundary lifting, pressure, reactions and restart."""
import numpy as np
import pytest
from physgraph.fem.flow_linear_solver import backend_identity
from physgraph.fem.navier_stokes import solve_navier_stokes
from physgraph.mesh.structured import tagged_unit_square_mesh
from physgraph.physics.navier_stokes import NavierStokesCase


def require_mumps():
    pytest.importorskip('petsc4py')
    from petsc4py import PETSc
    if not PETSc.Sys.hasExternalPackage('mumps'):
        pytest.skip('optional PETSc MUMPS unavailable')


def test_unknown_backend_rejected():
    with pytest.raises(ValueError, match='linear_backend'):
        backend_identity('automatic')


@pytest.mark.parametrize('traction', [False, True])
@pytest.mark.parametrize('scheme', ['be', 'bdf2'])
def test_mumps_nonzero_boundary_fields_and_restart(traction, scheme):
    require_mumps()
    case = NavierStokesCase('translated_vortex', pressure_offset=2.5 if traction else 0.)
    mesh = tagged_unit_square_mesh(3)
    def run(backend, steps=4, state=None):
        return solve_navier_stokes(mesh, case.forcing, case.velocity,
            lambda x: case.velocity(x, 0.), dt=.03, steps=steps, scheme=scheme,
            dirichlet_tags={'left','top','bottom'} if traction else None,
            tractions={'right':case.traction} if traction else None,
            problem_id='translated-vortex:v1', linear_backend=backend, restart_state=state)
    reference = run('scipy')
    candidate = run('mumps')
    for name in ('velocity', 'pressure', 'reactions'):
        np.testing.assert_allclose(getattr(candidate,name), getattr(reference,name), atol=1e-9, rtol=0)
    assert max(d['max_boundary_error'] for d in candidate.diagnostics) == 0.
    assert max(d['relative_residual'] for d in candidate.diagnostics) < 1e-12
    assert max(d['algebraic_force_balance'] for d in candidate.diagnostics) < 1e-9
    first = run('mumps', steps=2)
    resumed = run('mumps', steps=2, state=first.restart_state)
    for name in ('velocity', 'pressure', 'reactions'):
        np.testing.assert_array_equal(getattr(resumed,name), getattr(candidate,name)[2:])
    with pytest.raises(ValueError, match='restart contract'):
        run('scipy', steps=2, state=first.restart_state)
