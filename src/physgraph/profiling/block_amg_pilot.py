"""Bounded CPU convergence pilot before porting a block preconditioner to GPU."""
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.sparse import load_npz
from scipy.sparse.linalg import LinearOperator,gmres
from physgraph.fem.block_amg import BlockAMG,load_pyamg
from physgraph.io.manifest import write_manifest
from physgraph.profiling.flow_linear import file_sha256


def run_pilot(*,fixtures,output):
    root,dest=Path(fixtures),Path(output)
    if dest.exists() and any(dest.iterdir()):raise ValueError('output must be empty')
    protocol=json.loads((root/'manifest.json').read_text())
    config=dict(fixtures=str(root.resolve()),fixture_sha256=file_sha256(root/'manifest.json'),
        pyamg=load_pyamg().__version__,rtol=1e-12,restart=40,max_iterations=160,
        scope='CPU algebraic convergence pilot, not a GPU speedup measurement')
    rows=[];start=perf_counter()
    for case in protocol['metrics']['cases']:
        row=dict(case=case['path'],shape=case['shape'],nnz=case['nnz'])
        try:
            folder=root/case['path'];a=load_npz(folder/'matrix.npz')
            for name in ('matrix','vectors'):
                if file_sha256(folder/(name+'.npz'))!=case[name+'_sha256']:raise ValueError('fixture checksum mismatch')
            with np.load(folder/'vectors.npz') as data:b,ref=data['rhs'],data['reference']
            tick=perf_counter();pc=BlockAMG(a);row['setup_seconds']=perf_counter()-tick
            row['hierarchies']={'velocity':[l['A'].shape[0] for l in pc.velocity],
                                'pressure_factor_nnz':pc.pressure_factor.L.nnz+pc.pressure_factor.U.nnz}
            count=[];tick=perf_counter()
            def observe(value):
                count.append(float(value))
                if perf_counter()-tick>90:raise TimeoutError('pilot iteration budget exceeded')
            x,info=gmres(a,b,M=LinearOperator(a.shape,matvec=pc.apply),rtol=1e-12,atol=0.,
                restart=40,maxiter=4,callback=observe,callback_type='pr_norm')
            row.update(solve_seconds=perf_counter()-tick,iterations=len(count),info=info,
                       reference_error=float(np.max(np.abs(x-ref))),residual=float(np.linalg.norm(a@x-b)/max(1.,np.linalg.norm(b))))
            row['passed']=bool(info==0 and row['reference_error']<=1e-9 and row['residual']<=1e-10)
        except Exception as exc:row.update(passed=False,reason=str(exc))
        rows.append(row)
        write_manifest(dest/'manifest.json',config,dict(cases=rows),perf_counter()-start)
        print(json.dumps(row),flush=True)
    return rows
