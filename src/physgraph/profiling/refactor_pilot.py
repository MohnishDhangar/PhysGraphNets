"""Bounded, synchronized sparse refactorization pilot; no qualification launch."""
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.sparse import load_npz
from scipy.sparse.linalg import splu
from physgraph.fem.rocm_flow_solver import rocm_identity
from physgraph.fem.rocm_refactor import RocmRefactorSolver
from physgraph.io.manifest import write_manifest
from physgraph.profiling.flow_linear import file_sha256


def run_refactor_pilot(*, fixtures, output, case):
    root, dest = Path(fixtures), Path(output)
    if dest.exists() and any(dest.iterdir()): raise ValueError('output must be empty')
    protocol = json.loads((root/'manifest.json').read_text())
    fixture = next(row for row in protocol['metrics']['cases'] if row['path'] == case)
    runtime = rocm_identity(); runtime['name'] = 'rocsolver_refactor_pilot'; runtime.pop('settings')
    config = dict(runtime=runtime,case=case,shape=fixture['shape'],nnz=fixture['nnz'],
        fixture_manifest_sha256=file_sha256(root/'manifest.json'),
        scope='fixed-pattern fixture and deterministic value perturbations; not trajectory performance',
        precision='float64',timing='perf_counter, synchronized stages; includes CPU initial LU and transfers; sequential CPU controls')
    metrics = dict(status='running',cases=[])
    start = perf_counter()
    def save():
        report = write_manifest(dest/'manifest.json',config,metrics,perf_counter()-start)
        report['device'] = 'cpu_initial_lu_gpu_refactor_and_solve'
        (dest/'manifest.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        return report
    save()
    solver = None
    try:
        folder = root/case
        for name in ('matrix','vectors'):
            if file_sha256(folder/(name+'.npz')) != fixture[name+'_sha256']:
                raise ValueError('fixture checksum mismatch')
        a = load_npz(folder/'matrix.npz')
        with np.load(folder/'vectors.npz',allow_pickle=False) as values:
            b, archived = values['rhs'], values['reference']
        print('building',case,flush=True)
        solver = RocmRefactorSolver(a)
        metrics.update(cpu_initial_setup_seconds=solver.cpu_setup_seconds,
                       gpu_initial_setup_seconds=solver.gpu_setup_seconds)
        save()
        for shift in (0.,1e-4,-1e-4):
            current = a.copy()
            # Preserve sparsity while varying each nonzero, not merely scaling A.
            current.data *= 1 + shift*np.sin(np.arange(current.nnz,dtype=float))
            tick = perf_counter(); reference = splu(current.tocsc()).solve(b)
            cpu_seconds = perf_counter()-tick
            x = solver(current,b)
            error = float(np.max(np.abs(x-reference)))
            row = dict(shift=shift,reference_error=error,cpu_factor_solve_seconds=cpu_seconds,
                       **solver.records[-1])
            if shift == 0.: row['archive_error'] = float(np.max(np.abs(x-archived)))
            row['passed'] = bool(row['passed'] and error<=1e-9 and row.get('archive_error',0.)<=1e-9)
            metrics['cases'].append(row)
            metrics['memory'] = solver.gpu.memory_report()
            save(); print(json.dumps(row),flush=True)
        metrics['status'] = 'passed' if all(r['passed'] for r in metrics['cases']) else 'failed_numerical_check'
    except Exception as exc:
        metrics.update(status='failed',reason=f'{type(exc).__name__}: {exc}')
        if solver is not None:
            metrics['linear_records'] = solver.records
            metrics['memory'] = solver.gpu.memory_report()
    finally:
        if solver is not None: solver.close()
    return save()
