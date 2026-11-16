"""Exact temporal continuation, durable state, and incompatible restart rejection."""
from copy import deepcopy
import json
import numpy as np
import pytest
from physgraph.fem.navier_stokes import solve_navier_stokes
from physgraph.mesh.structured import tagged_unit_square_mesh
from physgraph.physics.navier_stokes import NavierStokesCase
from physgraph.io.flow_checkpoint import load_flow_checkpoint, save_flow_checkpoint


def run(*, steps=4, scheme='bdf2', traction=False, name='translated_vortex', **kwargs):
    case = NavierStokesCase(name, pressure_offset=2.5 if traction else 0.)
    initial_time = kwargs.get('initial_time', .13)
    settings = dict(dt=.03, initial_time=initial_time, problem_id=f'{name}:exponential:v1',
                    viscosity=case.viscosity)
    settings.update(kwargs)
    mesh = settings.pop('mesh', tagged_unit_square_mesh(2))
    return solve_navier_stokes(mesh, case.forcing, case.velocity,
        lambda x: case.velocity(x, initial_time), steps=steps, scheme=scheme,
        dirichlet_tags={'left', 'top', 'bottom'} if traction else None,
        tractions={'right': case.traction} if traction else None, **settings)


@pytest.mark.parametrize('scheme', ['be', 'bdf2'])
@pytest.mark.parametrize('split', [1, 3])
@pytest.mark.parametrize('traction', [False, True])
@pytest.mark.parametrize('name', ['translated_vortex', 'accelerating_shear'])
def test_saved_restart_is_bitwise_equal_to_uninterrupted(scheme, split, traction, name, tmp_path):
    options = dict(scheme=scheme, traction=traction, name=name)
    full = run(**options)
    first = run(steps=split, **options)
    path = tmp_path/'checkpoint.npz'
    save_flow_checkpoint(path, first.restart_state)
    state = load_flow_checkpoint(path)
    events = []
    resumed = run(steps=4-split, restart_state=state,
                  step_callback=lambda step, time: events.append((step, time)), **options)
    np.testing.assert_array_equal(resumed.times, full.times[split:])
    np.testing.assert_array_equal(resumed.velocity, full.velocity[split:])
    np.testing.assert_array_equal(resumed.pressure, full.pressure[split:])
    np.testing.assert_array_equal(resumed.reactions, full.reactions[split:])
    assert resumed.diagnostics == full.diagnostics[split:]
    assert resumed.initial_diagnostics == full.initial_diagnostics
    assert events == list(zip(range(split+1, 5), full.times[split+1:]))
    assert resumed.restart_state.step == 4
    np.testing.assert_array_equal(resumed.restart_state.older, full.restart_state.older)
    with pytest.raises(ValueError, match='pressure'):
        resumed.frame(0)


def test_callback_failure_leaves_resumable_atomic_checkpoint(tmp_path):
    path = tmp_path/'latest.npz'
    def stop(state):
        save_flow_checkpoint(path, state)
        if state.step == 2:
            raise InterruptedError('simulated process interruption after checkpoint')
    with pytest.raises(InterruptedError):
        run(checkpoint_callback=stop)
    assert list(tmp_path.iterdir()) == [path]
    state = load_flow_checkpoint(path)
    assert state.step == 2
    continued = run(steps=2, restart_state=state)
    full = run()
    np.testing.assert_array_equal(continued.velocity, full.velocity[2:])
    np.testing.assert_array_equal(continued.pressure, full.pressure[2:])


def test_checkpoint_callback_cannot_mutate_integrator():
    def corrupt(state):
        state.previous[:] = 9.
        state.older[:] = 8.
        state.guess[:] = 7.
        state.contract.clear()
        state.initial_diagnostics.clear()
    full, observed = run(), run(checkpoint_callback=corrupt)
    np.testing.assert_array_equal(full.velocity, observed.velocity)
    np.testing.assert_array_equal(full.pressure, observed.pressure)
    assert observed.restart_state.contract == full.restart_state.contract


@pytest.mark.parametrize('changes', [dict(dt=.02), dict(scheme='be'), dict(viscosity=.2),
    dict(initial_time=.14), dict(problem_id='changed-forcing:v2'), dict(quadrature_order=8),
    dict(newton_tolerance=1e-11), dict(max_iterations=21), dict(traction=True),
    dict(viscous_form='laplacian'), dict(mesh=tagged_unit_square_mesh(3))])
def test_incompatible_problem_or_discretization_rejected(changes):
    state = run(steps=1).restart_state
    with pytest.raises(ValueError):
        run(restart_state=state, **changes)


@pytest.mark.parametrize('kind', ['nan', 'shape', 'dtype', 'guess', 'boundary', 'older', 'step', 'source', 'runtime'])
def test_invalid_restart_state_rejected(kind):
    state = deepcopy(run(steps=2).restart_state)
    if kind == 'nan':
        state.older[0] = np.nan
    elif kind == 'shape':
        state.older = state.older[:-1]
    elif kind == 'dtype':
        state.previous = state.previous.astype(np.float32)
    elif kind == 'guess':
        state.guess[0] += 1.
    elif kind == 'boundary':
        state.guess[0] += 1.
        state.previous[0] += 1.
    elif kind == 'older':
        state.older[0] += 1.
    elif kind == 'step':
        state.step = True
    elif kind == 'source':
        state.contract['source'] = 'changed'
    else:
        state.contract['runtime']['numpy'] = 'changed'
    with pytest.raises(ValueError):
        run(restart_state=state)


def test_archive_integrity_and_source_rejection(tmp_path):
    path = tmp_path/'state.npz'
    state = run(steps=1).restart_state
    save_flow_checkpoint(path, state)
    with np.load(path, allow_pickle=False) as data:
        values = {key: data[key].copy() for key in data.files}
    values['older'][0] += 1.
    np.savez_compressed(path, **values)
    with pytest.raises(ValueError, match='checksum'):
        load_flow_checkpoint(path)
    state.contract['source'] = 'wrong-source'
    save_flow_checkpoint(path, state)
    with pytest.raises(ValueError, match='source'):
        load_flow_checkpoint(path)


def test_checkpoint_contract_requires_explicit_physics_identity():
    with pytest.raises(ValueError, match='problem_id'):
        run(problem_id=None, checkpoint_callback=lambda state: None)
    with pytest.raises(ValueError, match='problem_id'):
        run(problem_id=None, restart_state=run(steps=1).restart_state)
    with pytest.raises(ValueError, match='checkpoint_callback'):
        run(checkpoint_callback=True)
    assert run(steps=1, problem_id=None).restart_state is None


@pytest.mark.parametrize('scheme', ['be', 'bdf2'])
def test_resumed_closed_wall_energy_retains_pre_segment_history(scheme, tmp_path):
    from physgraph.evaluation.navier_stokes import closed_wall_energy
    from physgraph.experiments.navier_stokes import save_trajectory
    from physgraph.verification.navier_stokes import compare_navier_stokes
    mesh = tagged_unit_square_mesh(3)
    case = NavierStokesCase('vortex')
    zero = lambda x, t: np.zeros_like(x)
    def decay(steps, state=None):
        return solve_navier_stokes(mesh, zero, zero, lambda x: case.velocity(x, 0),
            dt=.03, steps=steps, scheme=scheme, problem_id='unforced-decay:v1', restart_state=state)
    full = decay(4)
    first = decay(2)
    resumed = decay(2, first.restart_state)
    reference = closed_wall_energy(full)
    actual = closed_wall_energy(resumed)
    assert actual['rows'] == reference['rows'][2:]
    assert actual['max_absolute_balance'] < 1e-10
    with pytest.raises(ValueError, match='complete trajectory'):
        compare_navier_stokes(resumed, case)
    path = tmp_path/'segment.npz'
    save_trajectory(path, resumed, {})
    with np.load(path, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive['velocity_before_segment'], full.velocity[1])
        np.testing.assert_array_equal(archive['pressure_times'], full.times[3:])
        assert json.loads(str(archive['metadata']))['segment_start_step'] == 2


def test_failed_write_preserves_previous_checkpoint(tmp_path, monkeypatch):
    path = tmp_path/'state.npz'
    first = run(steps=1).restart_state
    save_flow_checkpoint(path, first)
    original = path.read_bytes()
    def fail(stream, **kwargs):
        stream.write(b'incomplete archive')
        raise OSError('simulated disk write error')
    monkeypatch.setattr(np, 'savez_compressed', fail)
    with pytest.raises(OSError, match='disk write error'):
        save_flow_checkpoint(path, run(steps=2).restart_state)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]
    assert load_flow_checkpoint(path).step == 1
