"""Repeated complete startup/wake segments; all timings and failures retained."""
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
import numpy as np
from physgraph.io.manifest import write_manifest,environment_manifest
from physgraph.profiling.flow_linear import file_sha256
from physgraph.evaluation.gpu_flow_study import assess_gpu_study


def run_gpu_flow_study(*,seeds,output,repeats=3,baseline='mumps',candidate='rocm_gmres'):
    if type(repeats) is not int or repeats<3:
        raise ValueError('at least three repeats required')
    if baseline not in ('mumps','cpu_cached') or candidate not in ('rocm_gmres','rocm_cached'):
        raise ValueError('unsupported comparison backends')
    threads={'mumps':6,'rocm_gmres':6,'cpu_cached':1,'rocm_cached':1}
    root,dest=Path(seeds),Path(output)
    if dest.exists() and any(dest.iterdir()):
        raise ValueError('study output must be empty')
    protocol=json.loads((root/'manifest.json').read_text())
    cases=protocol['metrics']['cases']
    if not any(c['start_step']==0 for c in cases) or not any(c['start_step']>0 for c in cases):
        raise ValueError('both startup and developed-state seeds required')
    for case in cases:
        if file_sha256(root/case['path'])!=case['sha256']:
            raise ValueError('seed checksum mismatch')
    dest.mkdir(parents=True,exist_ok=True)
    script=Path(__file__).resolve().parents[3]/'scripts/qualify_gpu_solver.py'
    identity=environment_manifest()['source_sha256']
    config=dict(seeds=str(root.resolve()),seed_manifest_sha256=file_sha256(root/'manifest.json'),
        repeats=repeats,backends=[baseline,candidate],requested_cpu_threads={b:threads[b] for b in (baseline,candidate)},
        minimum_end_to_end_speedup=1.5,field_and_force_tolerance=1e-9,
        timing='sequential fresh subprocesses; entire wall time includes imports, seed read, FE assembly, preconditioner setup, transfers, Newton, output and evaluation; alternating order per repeat',
        launch_source_sha256=identity,scope='bounded startup and developed-wake GPU candidate, not full physical qualification')
    metrics=dict(status='running',runs=[],comparisons=[])
    start=perf_counter()
    def save():
        return write_manifest(dest/'manifest.json',config,metrics,perf_counter()-start)
    save()
    for repeat in range(repeats):
        for case in cases:
            for backend in ([baseline,candidate] if repeat%2==0 else [candidate,baseline]):
                label=f'{case["case"]}_{backend}_r{repeat}'
                folder=dest/label
                row=dict(case=case['case'],backend=backend,repeat=repeat,path=label)
                tick=perf_counter()
                try:
                    with (dest/(label+'.log')).open('w') as log:
                        completed=subprocess.run([sys.executable,str(script),'segment','--seed',str(root/case['path']),
                            '--output',str(folder),'--backend',backend],stdout=log,stderr=subprocess.STDOUT,
                            env={**os.environ,'OPENBLAS_NUM_THREADS':str(threads[backend]),'OMP_NUM_THREADS':str(threads[backend]),'MKL_NUM_THREADS':str(threads[backend]),'UCX_TLS':'self'},timeout=600)
                    row['process_wall_seconds']=perf_counter()-tick
                    if completed.returncode:
                        raise RuntimeError(f'worker exit {completed.returncode}; see {label}.log')
                    report=json.loads((folder/'manifest.json').read_text())
                    row.update(status=report['metrics']['status'],timings_seconds=report['metrics']['timings_seconds'],
                        peak_host_rss_bytes=report['metrics']['peak_host_rss_bytes'],
                        manifest_sha256=file_sha256(folder/'manifest.json'),source_sha256=report['source_sha256'])
                    if row['source_sha256']!=identity:
                        raise RuntimeError('source changed during worker execution')
                except (subprocess.SubprocessError,OSError,ValueError,RuntimeError) as exc:
                    row.update(status='failed',reason=str(exc),process_wall_seconds=perf_counter()-tick)
                metrics['runs'].append(row);save()
                print(label,row['status'],round(row['process_wall_seconds'],3),flush=True)
            pair=[r for r in metrics['runs'] if r['case']==case['case'] and r['repeat']==repeat]
            comparison=dict(case=case['case'],repeat=repeat,passed=False)
            if len(pair)==2 and all(r['status']=='passed' for r in pair):
                directories={r['backend']:dest/r['path'] for r in pair}
                with np.load(directories[baseline]/'trajectory.npz',allow_pickle=False) as cpu,np.load(
                    directories[candidate]/'trajectory.npz',allow_pickle=False) as gpu:
                    errors={key:float(np.max(np.abs(cpu[key]-gpu[key]))) for key in ('velocity','pressure','velocity_reactions')}
                    same_times=np.array_equal(cpu['times'],gpu['times'])
                histories={b:json.loads((p/'manifest.json').read_text())['metrics']['history']['rows'] for b,p in directories.items()}
                for key in ('drag','lift','pressure_difference','surface_drag','surface_lift'):
                    errors[key]=float(np.max(np.abs(np.array([r[key] for r in histories[baseline]])-
                                                       np.array([r[key] for r in histories[candidate]]))))
                comparison.update(max_absolute_errors=errors,passed=bool(same_times and all(np.isfinite(v) and v<=1e-9 for v in errors.values())))
            metrics['comparisons'].append(comparison);save()
    metrics['assessment']=assess_gpu_study(metrics['runs'],metrics['comparisons'],[c['case'] for c in cases],repeats=repeats,baseline=baseline,candidate=candidate)
    metrics['source_unchanged']=environment_manifest()['source_sha256']==identity
    metrics['status']=('failed_numerical_or_source_checks' if not metrics['source_unchanged'] or not metrics['assessment']['numerically_valid']
        else 'candidate_accepted' if metrics['assessment']['adopt'] else 'candidate_rejected_performance')
    return save()
