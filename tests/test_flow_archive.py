"""Bounded trajectory storage, numerical continuity, and transactional recovery."""
from copy import deepcopy
import subprocess
import sys
from pathlib import Path
import weakref
import numpy as np
import pytest
from physgraph.fem.navier_stokes import solve_navier_stokes
from physgraph.experiments.flow_stream import stream_navier_stokes
from physgraph.io.flow_archive import FlowArchive
from physgraph.mesh.structured import tagged_unit_square_mesh, unit_square_mesh
from physgraph.physics.navier_stokes import NavierStokesCase


def setup(scheme='bdf2', traction=False, tagged=True):
    case = NavierStokesCase('translated_vortex', pressure_offset=2.5 if traction else 0.)
    mesh = tagged_unit_square_mesh(2) if tagged else unit_square_mesh(2, 2)
    args = (mesh, case.forcing, case.velocity, lambda x: case.velocity(x, .13))
    kwargs = dict(dt=.03, initial_time=.13, viscosity=case.viscosity, scheme=scheme,
                  problem_id='translated_vortex:exp:offset='+str(case.pressure_offset),
                  dirichlet_tags={'left', 'top', 'bottom'} if traction else None,
                  tractions={'right':case.traction} if traction else None)
    return args, kwargs


def assert_matches(archive, full):
    start = 0
    for segment in archive.iter_segments():
        end = start+len(segment.times)-1
        for name in ('times', 'velocity'):
            np.testing.assert_array_equal(getattr(segment, name), getattr(full, name)[start:end+1])
        for name in ('pressure', 'reactions'):
            np.testing.assert_array_equal(getattr(segment, name), getattr(full, name)[start:end])
        assert segment.diagnostics == full.diagnostics[start:end]
        assert segment.initial_diagnostics == full.initial_diagnostics
        if start:
            np.testing.assert_array_equal(segment.velocity_before_segment, full.velocity[start-1])
        start = end
    assert start == len(full.times)-1
    assert archive.validate()['completed_steps'] == start


@pytest.mark.parametrize('scheme', ['be', 'bdf2'])
@pytest.mark.parametrize('traction', [False, True])
@pytest.mark.parametrize('chunk_steps', [1, 2, 4])
def test_streamed_and_extended_archive_matches_uninterrupted(tmp_path, scheme, traction, chunk_steps):
    args, options = setup(scheme, traction)
    full = solve_navier_stokes(*args, steps=7, **options)
    events = []
    archive = stream_navier_stokes(*args, output=tmp_path, total_steps=3, chunk_steps=chunk_steps,
                                   chunk_callback=events.append, **options)
    assert events[-1]['completed_steps'] == 3
    snapshot = archive
    archive = stream_navier_stokes(*args, output=tmp_path, total_steps=7, chunk_steps=chunk_steps,
                                   resume=True, **options)
    assert snapshot.completed_steps == 3  # reader sees its immutable committed prefix
    assert_matches(archive, full)
    assert archive.restart_state().step == 7
    # Complete resume is a validated no-op, not another solve.
    before = {p.name: p.stat().st_mtime_ns for p in tmp_path.iterdir() if p.name != 'writer.lock'}
    stream_navier_stokes(*args, output=tmp_path, total_steps=7, resume=True, **options)
    after = {p.name: p.stat().st_mtime_ns for p in tmp_path.iterdir() if p.name != 'writer.lock'}
    assert before == after


def test_untagged_mesh_roundtrip_without_pickle(tmp_path):
    args, options = setup(tagged=False)
    archive = stream_navier_stokes(*args, output=tmp_path, total_steps=2, chunk_steps=1, **options)
    assert archive.space.mesh.boundary_edges is None
    assert_matches(archive, solve_navier_stokes(*args, steps=2, **options))


def test_short_cylinder_stream_with_hole_and_laplacian_outlet(tmp_path):
    from physgraph.physics.cylinder_flow import CylinderFlowCase
    from physgraph.mesh.cylinder_flow import cylinder_flow_mesh
    case = CylinderFlowCase()
    mesh = cylinder_flow_mesh(case, 2)
    args = (mesh, case.zero, case.boundary_velocity, case.zero)
    options = dict(dt=.02, viscosity=case.viscosity, dirichlet_tags=case.dirichlet_tags,
                   tractions=case.tractions, viscous_form='laplacian', quadrature_order=4,
                   problem_id='dfg_2d3:default:v1')
    full = solve_navier_stokes(*args, steps=5, **options)
    stream_navier_stokes(*args, output=tmp_path, total_steps=2, chunk_steps=1, **options)
    archive = stream_navier_stokes(*args, output=tmp_path, total_steps=5, chunk_steps=2,
                                   resume=True, **options)
    assert_matches(archive, full)


def test_failure_inside_segment_restarts_at_last_commit(tmp_path):
    args, options = setup()
    events = []
    def stop(step, time):
        events.append(step)
        if step == 3:
            raise InterruptedError('incomplete chunk')
    with pytest.raises(InterruptedError):
        stream_navier_stokes(*args, output=tmp_path, total_steps=5, chunk_steps=2,
                            step_callback=stop, **options)
    assert FlowArchive(tmp_path).completed_steps == 2
    assert events == [1, 2, 3]
    archive = stream_navier_stokes(*args, output=tmp_path, total_steps=5, chunk_steps=2,
                                   resume=True, **options)
    assert_matches(archive, solve_navier_stokes(*args, steps=5, **options))


def test_abrupt_process_exit_releases_writer_and_preserves_committed_prefix(tmp_path):
    script = '''
import os
import sys
from physgraph.experiments.flow_stream import stream_navier_stokes
from physgraph.mesh.structured import tagged_unit_square_mesh
from physgraph.physics.navier_stokes import NavierStokesCase
case = NavierStokesCase('translated_vortex')
stream_navier_stokes(tagged_unit_square_mesh(2), case.forcing, case.velocity,
    lambda x:case.velocity(x,.13), output=sys.argv[1], total_steps=5, chunk_steps=2,
    dt=.03, initial_time=.13, problem_id='translated_vortex:exp:offset='+str(case.pressure_offset),
    chunk_callback=lambda event:os._exit(17))
'''
    process = subprocess.run([sys.executable, '-c', script, str(tmp_path)],
                             capture_output=True, text=True, timeout=30)
    assert process.returncode == 17, process.stderr
    assert FlowArchive(tmp_path).completed_steps == 2
    args, options = setup()
    archive = stream_navier_stokes(*args, output=tmp_path, total_steps=5, chunk_steps=2,
                                   resume=True, **options)
    assert_matches(archive, solve_navier_stokes(*args, steps=5, **options))


@pytest.mark.parametrize('failure_point', ['checkpoint', 'publication'])
def test_uncommitted_artifacts_are_not_replayed(tmp_path, monkeypatch, failure_point):
    import physgraph.io.flow_archive as module
    args, options = setup()
    stream_navier_stokes(*args, output=tmp_path, total_steps=2, chunk_steps=2, **options)
    index_bytes = (tmp_path/'index.json').read_bytes()
    with monkeypatch.context() as patch:
        def fail(*args, **kwargs):
            raise OSError('simulated commit failure')
        if failure_point == 'checkpoint':
            patch.setattr(module, 'save_flow_checkpoint', fail)
        else:
            patch.setattr(FlowArchive, '_publish', staticmethod(fail))
        with pytest.raises(OSError, match='commit failure'):
            stream_navier_stokes(*args, output=tmp_path, total_steps=5, chunk_steps=2,
                                resume=True, **options)
    assert (tmp_path/'index.json').read_bytes() == index_bytes
    assert (tmp_path/'segment_00000002.npz').exists()
    assert FlowArchive(tmp_path).validate()['completed_steps'] == 2
    archive = stream_navier_stokes(*args, output=tmp_path, total_steps=5, chunk_steps=1,
                                   resume=True, **options)
    assert_matches(archive, solve_navier_stokes(*args, steps=5, **options))


def test_exclusive_writer_lock_releases_on_exception(tmp_path):
    with pytest.raises(InterruptedError):
        with FlowArchive.writer(tmp_path):
            with pytest.raises(RuntimeError, match='another process'):
                with FlowArchive.writer(tmp_path, resume=True):
                    pass
            raise InterruptedError()
    with FlowArchive.writer(tmp_path, resume=True) as archive:
        assert archive.completed_steps == 0


def test_uncertain_publication_requires_reopen_before_further_append(tmp_path, monkeypatch):
    args, options = setup()
    first = solve_navier_stokes(*args, steps=2, **options)
    publish = FlowArchive._publish
    with FlowArchive.writer(tmp_path) as writer:
        def fail_after_publish(path, index):
            publish(path, index)
            raise OSError('simulated directory sync failure after rename')
        with monkeypatch.context() as patch:
            patch.setattr(FlowArchive, '_publish', staticmethod(fail_after_publish))
            with pytest.raises(OSError, match='sync failure'):
                writer.append(first)
        with pytest.raises(RuntimeError, match='writer context'):
            writer.append(first)
    assert FlowArchive(tmp_path).validate()['completed_steps'] == 2
    archive = stream_navier_stokes(*args, output=tmp_path, total_steps=4, resume=True, **options)
    assert_matches(archive, solve_navier_stokes(*args, steps=4, **options))


@pytest.mark.parametrize('target', ['mesh.npz','segment_00000001.npz','state_00000002.npz'])
def test_corrupt_committed_data_rejected(tmp_path, target):
    args, options = setup()
    stream_navier_stokes(*args, output=tmp_path, total_steps=4, chunk_steps=2, **options)
    path = tmp_path/target
    with np.load(path, allow_pickle=False) as archive:
        values = {k:archive[k].copy() for k in archive.files}
    key = 'nodes' if target == 'mesh.npz' else 'velocity' if target.startswith('segment') else 'older'
    values[key].flat[0] += .1
    np.savez_compressed(path, **values)
    with pytest.raises(ValueError, match='checksum'):
        stream_navier_stokes(*args, output=tmp_path, total_steps=6, resume=True, **options)


@pytest.mark.parametrize('changed', [dict(dt=.02), dict(problem_id='different'), dict(scheme='be')])
def test_complete_resume_still_checks_scientific_contract(tmp_path, changed):
    args, options = setup()
    stream_navier_stokes(*args, output=tmp_path, total_steps=2, **options)
    with pytest.raises(ValueError, match='contract'):
        stream_navier_stokes(*args, output=tmp_path, total_steps=2, resume=True, **{**options, **changed})


def test_memory_release_and_chunk_bound_independent_of_total_steps(tmp_path, monkeypatch):
    import physgraph.experiments.flow_stream as module
    original = module.solve_navier_stokes
    references, sizes = [], []
    # Preserve signature: the orchestrator uses it to resolve default contracts.
    from functools import wraps
    @wraps(original)
    def observe(*args, **kwargs):
        assert all(ref() is None for ref in references)
        result = original(*args, **kwargs)
        sizes.append(len(result.velocity))
        references.append(weakref.ref(result.velocity))
        return result
    monkeypatch.setattr(module, 'solve_navier_stokes', observe)
    args, options = setup()
    archive = module.stream_navier_stokes(*args, output=tmp_path, total_steps=13, chunk_steps=2, **options)
    assert max(sizes) == 3
    assert len(sizes) == 7
    assert all(ref() is None for ref in references)
    assert archive.completed_steps == 13


def test_source_drift_cannot_publish_a_segment(tmp_path, monkeypatch):
    import physgraph.experiments.flow_stream as module
    monkeypatch.setattr(module, 'flow_source_fingerprint', lambda:'changed-source')
    args, options = setup()
    with pytest.raises(RuntimeError, match='source changed'):
        module.stream_navier_stokes(*args, output=tmp_path, total_steps=2, **options)
    assert FlowArchive(tmp_path).completed_steps == 0


def test_archive_requires_explicit_writer_and_rejects_nonempty_destination(tmp_path):
    args, options = setup()
    archive = stream_navier_stokes(*args, output=tmp_path, total_steps=2, **options)
    with pytest.raises(RuntimeError, match='writer context'):
        archive.append(solve_navier_stokes(*args, steps=2, **options))
    with pytest.raises(ValueError, match='empty'):
        stream_navier_stokes(*args, output=tmp_path, total_steps=2, **options)
    with pytest.raises(ValueError, match='precedes'):
        stream_navier_stokes(*args, output=tmp_path, total_steps=1, resume=True, **options)


@pytest.mark.parametrize('change', ['pressure', 'times', 'diagnostic_time', 'history'])
def test_inconsistent_segment_cannot_be_committed(tmp_path, change):
    args, options = setup()
    first = solve_navier_stokes(*args, steps=2, **options)
    second = deepcopy(solve_navier_stokes(*args, steps=2, restart_state=first.restart_state, **options))
    if change == 'pressure':
        second.pressure[-1,0] += .1
    elif change == 'times':
        second.times[1] += .01
    elif change == 'diagnostic_time':
        second.diagnostics[0]['time'] += .01
    else:
        second.velocity_before_segment[0,0] += .1
    with FlowArchive.writer(tmp_path) as archive:
        archive.append(first)
        with pytest.raises(ValueError):
            archive.append(second)
        assert archive.completed_steps == 2
    assert FlowArchive(tmp_path).validate()['completed_steps'] == 2


@pytest.mark.parametrize('kwargs', [dict(total_steps=0), dict(chunk_steps=True), dict(resume='yes'),
    dict(problem_id=None), dict(checkpoint_callback=lambda x:None), dict(steps=1), dict(chunk_callback=True)])
def test_invalid_stream_configuration(tmp_path, kwargs):
    args, options = setup()
    with pytest.raises(ValueError):
        stream_navier_stokes(*args, output=tmp_path, **{**options, 'total_steps':2, **kwargs})


def test_cli_stream_and_resume_records_provenance_and_exact_patch_errors(tmp_path):
    script = Path(__file__).resolve().parents[1]/'scripts/run_flow_stream.py'
    command = [sys.executable, str(script), '--output', str(tmp_path), '--n', '2',
               '--chunk-steps', '1', '--traction-tag', 'right']
    first = subprocess.run(command+['--total-steps', '2'], capture_output=True, text=True, timeout=30)
    assert first.returncode == 0, first.stderr
    origin = (tmp_path/'origin_manifest.json').read_bytes()
    second = subprocess.run(command+['--total-steps', '4', '--resume'],
                            capture_output=True, text=True, timeout=30)
    assert second.returncode == 0, second.stderr
    import json
    report = json.loads((tmp_path/'manifest.json').read_text())
    assert report['metrics']['completed_steps'] == 4
    assert report['metrics']['checks']['exact_patch']
    assert max(report['metrics']['final_errors'].values()) < 1e-9
    assert report['python'] and report['source_sha256']
    assert (tmp_path/'origin_manifest.json').read_bytes() == origin
    assert FlowArchive(tmp_path/'trajectory').validate()['completed_steps'] == 4
