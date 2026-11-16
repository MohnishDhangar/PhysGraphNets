"""Actual GPU linear solves on declared fine-mesh systems with preserved failures."""
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.sparse import load_npz
from physgraph.fem.rocm_flow_solver import RocmFlowSolver,rocm_identity
from physgraph.io.manifest import write_manifest
from physgraph.profiling.flow_linear import file_sha256


def run_gpu_solver_checks(*,fixtures,output):
    root,dest=Path(fixtures),Path(output)
    if dest.exists() and any(dest.iterdir()):
        raise ValueError('output must be empty')
    protocol=json.loads((root/'manifest.json').read_text())
    config=dict(backend=rocm_identity(),fixture_manifest_sha256=file_sha256(root/'manifest.json'),
        fixture_manifest=str((root/'manifest.json').resolve()),reference_tolerance=1e-9,
        residual_tolerance=1e-10,scope='full linear solves; CPU ILUT setup plus GPU-resident Krylov and triangular operations')
    metrics=dict(status='running',cases=[])
    start=perf_counter()
    def save():
        report=write_manifest(dest/'manifest.json',config,metrics,perf_counter()-start)
        report['device']='hybrid_cpu_setup_gpu_iterations'
        (dest/'manifest.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        return report
    save()
    for case in protocol['metrics']['cases']:
        row=dict(case=case['path'],shape=case['shape'],nnz=case['nnz'])
        solver=RocmFlowSolver()
        try:
            folder=root/case['path']
            for name in ('matrix','vectors'):
                if file_sha256(folder/(name+'.npz'))!=case[name+'_sha256']:
                    raise ValueError('fixture checksum mismatch')
            a=load_npz(folder/'matrix.npz')
            with np.load(folder/'vectors.npz',allow_pickle=False) as archive:
                b,reference=archive['rhs'],archive['reference']
            x=solver(a,b)
            error=float(np.max(np.abs(x-reference)))
            residual=float(np.linalg.norm(a@x-b)/max(np.linalg.norm(b),1.))
            row.update(status='passed' if error<=1e-9 and residual<=1e-10 else 'failed_numerical_check',
                max_absolute_difference=error,relative_residual=residual,linear_records=solver.records)
        except Exception as exc:
            row.update(status='failed',reason=f'{type(exc).__name__}: {exc}')
        finally:
            solver.close()
        metrics['cases'].append(row);save()
        print(case['path'],row['status'],row.get('reason',''),flush=True)
    metrics['status']='passed' if all(r['status']=='passed' for r in metrics['cases']) else 'failed'
    return save()
