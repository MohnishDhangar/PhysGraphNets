"""DFG cylinder trajectories with explicit discrete and published-reference gates."""
from dataclasses import asdict
from contextlib import contextmanager
import csv
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from physgraph.physics.cylinder_flow import CylinderFlowCase
from physgraph.mesh.cylinder_flow import cylinder_flow_mesh
from physgraph.mesh.quality import mesh_quality
from physgraph.fem.navier_stokes import solve_navier_stokes
from physgraph.evaluation.cylinder_flow import cylinder_history, published_comparison
from physgraph.experiments.navier_stokes import save_trajectory
from physgraph.io.manifest import environment_manifest, write_manifest
from physgraph.fem.flow_linear_solver import backend_identity


def run_cylinder_flow(*, output, level=4, dt=.02, final_time=8., external=False, case=None,
                      linear_backend='scipy'):
    case = CylinderFlowCase() if case is None else case
    if not np.isfinite([dt,final_time]).all() or dt<=0 or not 0<final_time<=case.duration:
        raise ValueError('require positive dt and final time within case duration')
    steps=int(round(final_time/dt))
    if steps<1 or not np.isclose(steps*dt,final_time,atol=1e-12,rtol=0):
        raise ValueError('final time must be an integer multiple of dt')
    dest=Path(output)
    if dest.exists() and (not dest.is_dir() or any(dest.iterdir())):
        raise ValueError('cylinder output must be empty')
    dest.mkdir(parents=True,exist_ok=True)
    gates=dict(residual=5e-11,boundary=1e-12,flux=1e-10,momentum=1e-9,external=1e-9)
    config=dict(case=asdict(case),level=level,dt=dt,steps=steps,final_time=final_time,
        external=external,linear_backend=backend_identity(linear_backend),
        viscous_form='laplacian',outlet='zero Laplacian conormal',
        scheme='BDF2 with BE startup',quadrature_order=4,acceptance=gates,
        benchmark_acceptance={'relative_quantities':.03,'absolute_peak_times':.05},
        qualification_note='A single passing trajectory is a candidate; spatial/time refinement remains required')
    started=perf_counter()
    launch_identity=environment_manifest()
    config['launch_source_sha256']=launch_identity['source_sha256']
    metrics={'status':'running', 'timings_seconds':{}}
    @contextmanager
    def timed_phase(name):
        beginning=perf_counter()
        try:
            yield
        finally:
            metrics['timings_seconds'][name]=perf_counter()-beginning
    def progress(stage, step=0, time=0.):
        data=dict(stage=stage,step=step,steps=steps,time=time,final_time=final_time,
                  elapsed_seconds=perf_counter()-started)
        temporary=dest/'progress.tmp'
        temporary.write_text(json.dumps(data,allow_nan=False)+'\n')
        temporary.replace(dest/'progress.json')
    def solved(step, time):
        if step == 1 or step % max(1,steps//100) == 0 or step == steps:
            progress('solving',step,time)
    write_manifest(dest/'manifest.json',config,metrics,0.)
    try:
        progress('meshing')
        with timed_phase('meshing'):
            mesh=cylinder_flow_mesh(case,level)
            metrics['mesh_quality']=mesh_quality(mesh)
        with timed_phase('local_solve'):
            trajectory=solve_navier_stokes(mesh,case.zero,case.boundary_velocity,case.zero,
                viscosity=case.viscosity,dt=dt,steps=steps,dirichlet_tags=case.dirichlet_tags,
                tractions=case.tractions,viscous_form='laplacian',quadrature_order=4,
                step_callback=solved,linear_backend=linear_backend)
        if trajectory.linear_solver_records is not None:
            metrics['linear_solver_records']=trajectory.linear_solver_records
        progress('saving_trajectory',steps,final_time)
        with timed_phase('archive_write'):
            save_trajectory(dest/'trajectory.npz',trajectory,config)
        progress('evaluating',steps,final_time)
        with timed_phase('force_and_pressure_evaluation'):
            metrics['history']=cylinder_history(trajectory,case)
        rows=metrics['history']['rows']
        metrics['checks']=dict(residual=max(r['relative_residual'] for r in rows)<=gates['residual'],
            initial_projection=trajectory.initial_diagnostics['projection_residual']<=1e-10,
            weak_divergence=max(r['weak_divergence_residual'] for r in rows)<=1e-10,
            boundary=max(r['max_boundary_error'] for r in rows)<=gates['boundary'],
            no_slip=max(r['max_obstacle_speed'] for r in rows)<=gates['boundary'],
            flux=max(abs(r['net_flux']) for r in rows)<=gates['flux'],
            momentum=max(r['algebraic_force_balance'] for r in rows)<=gates['momentum'])
        if external:
            progress('independent_comparison',steps,final_time)
            from physgraph.verification.navier_stokes import compare_navier_stokes
            with timed_phase('independent_comparison'):
                metrics['fenicsx']=compare_navier_stokes(trajectory,case,gates['external'])
            metrics['checks']['fenicsx']=metrics['fenicsx']['passed']
        metrics['published']=published_comparison(metrics['history'],case,float(trajectory.times[-1]))
        metrics['checks']['source_unchanged']=(environment_manifest()['source_sha256']==launch_identity['source_sha256'])
        passed=all(metrics['checks'].values())
        metrics['status']=('failed_discrete_checks' if not passed else
            'failed_benchmark_acceptance' if metrics['published']['status']=='failed_acceptance' else
            'benchmark_candidate_passed' if metrics['published']['status']=='passed' else 'passed_discrete_checks')
        with (dest/'forces.csv').open('w',newline='') as stream:
            keys=['time','drag','lift','surface_drag','surface_lift','pressure_difference','net_flux']
            writer=csv.DictWriter(stream,fieldnames=keys)
            writer.writeheader()
            writer.writerows({k:r[k] for k in keys} for r in rows)
    except Exception as exc:
        metrics.update(status='failed',reason=f'{type(exc).__name__}: {exc}')
        progress('failed')
        write_manifest(dest/'manifest.json',config,metrics,perf_counter()-started)
        raise
    report=write_manifest(dest/'manifest.json',config,metrics,perf_counter()-started)
    progress(metrics['status'],steps,final_time)
    return report
