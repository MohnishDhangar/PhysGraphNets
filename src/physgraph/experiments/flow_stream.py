"""Chunked orchestration of the verified fixed-step flow integrator."""
from inspect import signature
from dataclasses import asdict
import json
from time import perf_counter
from physgraph.fem.navier_stokes import solve_navier_stokes, _restart_contract
from physgraph.fem.stokes import taylor_hood_space
from physgraph.fem.stokes_boundary import stokes_partition
from physgraph.io.flow_archive import FlowArchive, exclusive_directory_writer
from physgraph.io.manifest import environment_manifest, write_manifest
from physgraph.io.flow_checkpoint import flow_source_fingerprint


def stream_navier_stokes(mesh, forcing, boundary_velocity, initial_velocity, *, output,
                        total_steps, chunk_steps=100, resume=False, chunk_callback=None,
                        **solver_options):
    """Advance to an absolute total step count, committing bounded-size segments.

    `problem_id` and all ordinary solver options, including dt, belong in
    solver_options. The archive owns restart states and checkpoint publication.
    A step_callback observes converged steps; chunk_callback observes committed
    chunks. Failure discards only the uncommitted tail, at most chunk_steps steps.
    Memory is O(chunk_steps * spatial DOFs), independent of total_steps. Repeated
    assembly at segment boundaries is deliberate; no numerical history changes.
    """
    if type(total_steps) is not int or total_steps < 1 or type(chunk_steps) is not int or chunk_steps < 1:
        raise ValueError('total_steps and chunk_steps must be positive integers')
    if type(resume) is not bool:
        raise ValueError('resume must be a boolean')
    if chunk_callback is not None and not callable(chunk_callback):
        raise ValueError('chunk_callback must be callable')
    if {'steps', 'restart_state', 'checkpoint_callback'} & solver_options.keys():
        raise ValueError('streaming owns steps, restart state and checkpoint publication')
    if not isinstance(solver_options.get('problem_id'), str) or not solver_options['problem_id'].strip():
        raise ValueError('streaming requires an explicit nonempty problem_id')
    bound = signature(solve_navier_stokes).bind(mesh, forcing, boundary_velocity, initial_velocity,
                                               steps=1, **solver_options)
    bound.apply_defaults()
    options = bound.arguments
    space = taylor_hood_space(mesh)
    _, _, tractions, tags = stokes_partition(space, options['dirichlet_tags'], options['tractions'])
    contract = _restart_contract(space, tags, tractions, options)
    with FlowArchive.writer(output, resume=resume) as archive:
        if archive.completed_steps > total_steps:
            raise ValueError('total_steps precedes the committed archive')
        if archive.index['contract'] is not None and archive.index['contract'] != contract:
            raise ValueError('streaming restart contract mismatch')
        state = archive.restart_state()
        while archive.completed_steps < total_steps:
            count = min(chunk_steps, total_steps-archive.completed_steps)
            trajectory = solve_navier_stokes(mesh, forcing, boundary_velocity, initial_velocity,
                                            steps=count, restart_state=state, **solver_options)
            if trajectory.restart_state.contract != contract or flow_source_fingerprint() != contract['source']:
                raise RuntimeError('numerical source changed during streamed solve; segment not committed')
            archive.append(trajectory)
            state = trajectory.restart_state
            del trajectory
            if chunk_callback is not None:
                chunk_callback(dict(completed_steps=archive.completed_steps, total_steps=total_steps,
                                    committed_chunks=archive.index['chunks']))
    return FlowArchive(output)


def run_manufactured_flow_stream(*, output, n=4, name='accelerating_shear', time_profile='linear',
                                dt=.025, total_steps=16, chunk_steps=4, scheme='bdf2',
                                traction_tag=None, resume=False):
    """Reproducible CPU streaming experiment with bounded-memory error assessment.

    This checks discrete equations and records manufactured-reference errors.
    It is not a spatial/time refinement study or an obstacle-flow qualification.
    """
    from physgraph.mesh.structured import tagged_unit_square_mesh
    from physgraph.mesh.quality import mesh_quality
    from physgraph.physics.navier_stokes import NavierStokesCase
    from physgraph.evaluation.navier_stokes import trajectory_metrics
    case = NavierStokesCase(name, time_profile=time_profile,
                           pressure_offset=0. if traction_tag is None else 2.5)
    mesh = tagged_unit_square_mesh(n)
    if traction_tag not in (None, 'left', 'right', 'top', 'bottom'):
        raise ValueError('unknown traction boundary')
    gates = dict(relative_residual=5e-11, max_boundary_error=1e-12,
                 weak_divergence_residual=1e-10, net_integrated_flux=1e-10,
                 algebraic_force_balance=1e-9, initial_projection=1e-10, exact_patch=1e-9)
    config = dict(case=asdict(case), n=n, dt=dt, total_steps=total_steps,
                  chunk_steps=chunk_steps, scheme=scheme, traction_tag=traction_tag,
                  acceptance=gates, resume=resume,
                  scope='manufactured discrete checks and streamed storage; refinement not assessed')
    config['launch_source_sha256'] = environment_manifest()['source_sha256']
    started = perf_counter()
    with exclusive_directory_writer(output) as dest:
        if not resume and any(path.name != 'writer.lock' for path in dest.iterdir()):
            raise ValueError('new streamed experiment output must be empty')
        if resume and not (dest/'origin_manifest.json').exists():
            raise ValueError('resume requires the original experiment manifest')
        metrics = dict(status='running', mesh=mesh_quality(mesh), completed_steps=0)
        if not resume:
            write_manifest(dest/'origin_manifest.json', config, metrics, 0.)
        write_manifest(dest/'manifest.json', config, metrics, 0.)
        def committed(event):
            metrics.update(event)
            write_manifest(dest/'manifest.json', config, metrics, perf_counter()-started)
        try:
            archive = stream_navier_stokes(mesh, case.forcing, case.velocity, lambda x:case.velocity(x,0.),
                output=dest/'trajectory', total_steps=total_steps, chunk_steps=chunk_steps, resume=resume,
                chunk_callback=committed, dt=dt, viscosity=case.viscosity, scheme=scheme,
                problem_id='manufactured_flow:v1:'+json.dumps(asdict(case),sort_keys=True),
                dirichlet_tags=None if traction_tag is None else set(mesh.boundary_edge_tags)-{traction_tag},
                tractions=None if traction_tag is None else {traction_tag:case.traction})
            metrics.update(completed_steps=archive.completed_steps, committed_chunks=archive.index['chunks'])
            maxima, error_maxima = {}, {}
            for segment in archive.iter_segments():
                row = trajectory_metrics(segment, case)
                for key, value in row['maxima'].items():
                    maxima[key] = max(maxima.get(key, 0.), value)
                for key in row['final_errors']:
                    error_maxima[key] = max(error_maxima.get(key, 0.), max(frame[key] for frame in row['frames']))
                metrics['final_errors'] = row['final_errors']
                metrics['initial_projection'] = row['initial']['projection_residual']
            checks = {key: maxima[key] <= limit for key, limit in gates.items() if key in maxima}
            checks['initial_projection'] = metrics['initial_projection'] <= gates['initial_projection']
            if name == 'accelerating_shear' and time_profile == 'linear':
                checks['exact_patch'] = max(error_maxima.values()) <= gates['exact_patch']
            checks['source_unchanged'] = environment_manifest()['source_sha256'] == config['launch_source_sha256']
            metrics.update(maxima=maxima, maximum_reference_errors=error_maxima, checks=checks,
                           status='passed_discrete_checks' if all(checks.values()) else 'failed_discrete_checks')
        except Exception as exc:
            metrics.update(status='failed', reason=f'{type(exc).__name__}: {exc}')
            write_manifest(dest/'manifest.json', config, metrics, perf_counter()-started)
            raise
        return write_manifest(dest/'manifest.json', config, metrics, perf_counter()-started)
