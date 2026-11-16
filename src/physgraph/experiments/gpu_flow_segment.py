"""Run one complete bounded flow segment with a declared linear backend."""
from inspect import signature
import json
from pathlib import Path
import resource
from time import perf_counter
import numpy as np
from physgraph.core.types import Mesh2D
from physgraph.core.flow_state import FlowRestartState
from physgraph.fem.navier_stokes import solve_navier_stokes,_restart_contract
from physgraph.fem.stokes import taylor_hood_space
from physgraph.fem.stokes_boundary import stokes_partition
from physgraph.fem.flow_linear_solver import backend_identity
from physgraph.physics.cylinder_flow import CylinderFlowCase
from physgraph.evaluation.cylinder_flow import cylinder_history
from physgraph.experiments.navier_stokes import save_trajectory
from physgraph.mesh.quality import mesh_quality
from physgraph.profiling.flow_linear import file_sha256
from physgraph.io.manifest import write_manifest,environment_manifest


def run_flow_segment(*,seed,output,backend):
    started=perf_counter()
    source,dest=Path(seed),Path(output)
    if dest.exists() and any(dest.iterdir()):
        raise ValueError('segment output must be empty')
    identity=environment_manifest()['source_sha256']
    config=dict(seed=str(source.resolve()),seed_sha256=file_sha256(source),
        backend=backend_identity(backend),scope='bounded backend experiment; benchmark not assessed',
        launch_source_sha256=identity)
    metrics=dict(status='running',timings_seconds={})
    def save():
        metrics['peak_host_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        return write_manifest(dest/'manifest.json',config,metrics,perf_counter()-started)
    save()
    try:
        with np.load(source,allow_pickle=False) as archive:
            meta=json.loads(str(archive['metadata']))
            nodes,triangles=archive['nodes'],archive['triangles']
            edges,tags=archive['boundary_edges'],archive['boundary_edge_tags']
            previous,older,pressure=archive['previous'],archive['older'],archive['previous_pressure']
            expected={k:archive['expected_'+k] for k in ('velocity','pressure','reactions')}
            expected_times=archive['times']
        config['seed_metadata']=meta
        boundary=np.zeros(len(nodes),dtype=bool);boundary[np.unique(edges)]=True
        mesh=Mesh2D(nodes,triangles,boundary,edges,tags)
        metrics['mesh_quality']=mesh_quality(mesh)
        case_data=dict(meta['case']);case_data['center']=tuple(case_data['center'])
        case=CylinderFlowCase(**case_data)
        options=dict(dt=meta['dt'],steps=meta['steps'],viscosity=case.viscosity,
            scheme='bdf2',viscous_form='laplacian',quadrature_order=meta['quadrature_order'],
            dirichlet_tags=case.dirichlet_tags,tractions=case.tractions,linear_backend=backend,
            problem_id='archived-cylinder-seed:'+config['seed_sha256'])
        if meta['start_step']:
            # Construct a NEW comparison contract for common numerical input.
            # Never relabel an old qualification checkpoint as compatible.
            bound=signature(solve_navier_stokes).bind(mesh,case.zero,case.boundary_velocity,case.zero,**options)
            bound.apply_defaults()
            space=taylor_hood_space(mesh)
            _,_,traction_data,selected=stokes_partition(space,case.dirichlet_tags,case.tractions)
            contract=_restart_contract(space,selected,traction_data,bound.arguments)
            options['restart_state']=FlowRestartState(contract,meta['start_step'],previous.T.ravel().copy(),
                older.T.ravel().copy(),np.r_[previous.T.ravel(),pressure],
                dict(initial_projection_assessed=False,origin='archived completed CPU trajectory'))
        metrics['timings_seconds']['seed_load_and_setup']=perf_counter()-started
        tick=perf_counter()
        trajectory=solve_navier_stokes(mesh,case.zero,case.boundary_velocity,case.zero,**options)
        metrics['timings_seconds']['local_solve']=perf_counter()-tick
        tick=perf_counter()
        metrics['history']=cylinder_history(trajectory,case)
        metrics['timings_seconds']['evaluation']=perf_counter()-tick
        metrics['linear_solver_records']=trajectory.linear_solver_records
        metrics['archive_reference_errors']={k:float(np.max(np.abs(getattr(trajectory,k)-v))) for k,v in expected.items()}
        rows=metrics['history']['rows']
        metrics['checks']=dict(
            time_grid=bool(np.array_equal(trajectory.times,expected_times)),
            residual=max(r['relative_residual'] for r in rows)<=5e-11,
            boundary=max(r['max_boundary_error'] for r in rows)<=1e-12,
            no_slip=max(r['max_obstacle_speed'] for r in rows)<=1e-12,
            flux=max(abs(r['net_flux']) for r in rows)<=1e-10,
            weak_divergence=max(r['weak_divergence_residual'] for r in rows)<=1e-10,
            momentum=max(r['algebraic_force_balance'] for r in rows)<=1e-9,
            archived_fields=all(np.isfinite(v) and v<=1e-9 for v in metrics['archive_reference_errors'].values()))
        if meta['start_step']==0:
            metrics['checks']['initial_projection']=trajectory.initial_diagnostics['projection_residual']<=1e-10
        tick=perf_counter()
        save_trajectory(dest/'trajectory.npz',trajectory,config)
        metrics['timings_seconds']['archive_write']=perf_counter()-tick
        metrics['checks']['source_unchanged']=environment_manifest()['source_sha256']==identity
        metrics['status']='passed' if all(metrics['checks'].values()) else 'failed_checks'
    except Exception as exc:
        metrics.update(status='failed',reason=f'{type(exc).__name__}: {exc}')
        save()
        raise
    return save()
