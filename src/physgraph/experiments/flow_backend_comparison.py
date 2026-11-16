"""Bounded full-trajectory backend timing and field/load parity experiment."""
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
import numpy as np
from physgraph.io.manifest import write_manifest, environment_manifest


def run_backend_comparison(*, output, level=16, dt=.005, steps=6, repeats=3):
    if type(repeats) is not int or repeats < 2 or type(steps) is not int or steps < 1:
        raise ValueError('require >=2 repeats and positive integer steps')
    dest = Path(output)
    if dest.exists() and any(dest.iterdir()):
        raise ValueError('comparison output must be empty')
    dest.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve().parents[3]/'scripts/run_cylinder_flow.py'
    identity = environment_manifest()['source_sha256']
    config = dict(level=level, dt=dt, steps=steps, repeats=repeats,
        candidates={'scipy':1, 'mumps':6}, tolerance=1e-9,
        timing='fresh process; runner local_solve includes assembly/projection/all Newton work; subprocess wall also includes startup/archive/evaluation; alternating order by repeat',
        scope='bounded startup trajectory; not developed-wake or refinement qualification',
        launch_source_sha256=identity)
    metrics = dict(status='running', runs=[], comparisons=[])
    started = perf_counter()
    def save():
        return write_manifest(dest/'manifest.json', config, metrics, perf_counter()-started)
    save()
    try:
        for repeat in range(repeats):
            order = ['scipy','mumps'] if repeat%2 == 0 else ['mumps','scipy']
            for backend in order:
                count = config['candidates'][backend]
                folder = dest/f'{backend}_{repeat}'
                tick = perf_counter()
                with (dest/f'{backend}_{repeat}.log').open('w') as log:
                    completed = subprocess.run([sys.executable, str(script), '--output',str(folder),
                        '--level',str(level),'--dt',str(dt),'--final-time',str(dt*steps),
                        '--linear-backend',backend], stdout=log, stderr=subprocess.STDOUT,
                        env={**os.environ, 'OPENBLAS_NUM_THREADS':str(count), 'OMP_NUM_THREADS':str(count),
                             'MKL_NUM_THREADS':str(count), 'UCX_TLS':'self'}, timeout=600)
                wall = perf_counter()-tick
                if completed.returncode:
                    raise RuntimeError(f'{backend} repeat {repeat} failed: see log')
                report = json.loads((folder/'manifest.json').read_text())
                if not all(report['metrics']['checks'].values()):
                    raise RuntimeError('candidate failed discrete/source checks')
                metrics['runs'].append(dict(backend=backend,repeat=repeat,threads=count,
                    process_wall_seconds=wall,local_solve_seconds=report['metrics']['timings_seconds']['local_solve'],
                    source_sha256=report['source_sha256'], manifest=str(folder/'manifest.json')))
                save()
                print(f'{backend} repeat {repeat}: {wall:.3f}s',flush=True)
            with np.load(dest/f'scipy_{repeat}/trajectory.npz',allow_pickle=False) as ref, np.load(
                    dest/f'mumps_{repeat}/trajectory.npz',allow_pickle=False) as candidate:
                errors = {key:float(np.max(np.abs(candidate[key]-ref[key])))
                          for key in ('velocity','pressure','velocity_reactions')}
                if not np.array_equal(ref['times'],candidate['times']):
                    raise RuntimeError('time grids differ')
            reports = [json.loads((dest/f'{b}_{repeat}/manifest.json').read_text()) for b in ('scipy','mumps')]
            for key in ('drag','lift','pressure_difference','surface_drag','surface_lift'):
                values = [[r[key] for r in p['metrics']['history']['rows']] for p in reports]
                errors[key] = float(np.max(np.abs(np.asarray(values[0])-values[1])))
            metrics['comparisons'].append(dict(repeat=repeat,max_absolute_errors=errors,
                passed=all(np.isfinite(v) and v<=config['tolerance'] for v in errors.values())))
            save()
        metrics['summary'] = {backend:dict(
            median_local_solve_seconds=float(np.median([r['local_solve_seconds'] for r in metrics['runs'] if r['backend']==backend])),
            median_process_wall_seconds=float(np.median([r['process_wall_seconds'] for r in metrics['runs'] if r['backend']==backend])))
            for backend in ('scipy','mumps')}
        metrics['source_unchanged'] = environment_manifest()['source_sha256'] == identity
        metrics['status'] = 'passed' if metrics['source_unchanged'] and all(c['passed'] for c in metrics['comparisons']) else 'failed_parity_or_source'
    except Exception as exc:
        metrics.update(status='failed',reason=f'{type(exc).__name__}: {exc}')
        save()
        raise
    return save()
