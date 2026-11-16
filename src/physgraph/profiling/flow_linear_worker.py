"""Isolated float64 linear-backend trial with conversion-inclusive wall timing."""
import json
import os
from pathlib import Path
import resource
import sys
from time import perf_counter, process_time
import numpy as np
from scipy.sparse import load_npz
from scipy.sparse.linalg import spsolve


def trial(settings):
    folder = Path(settings['fixture'])
    a = load_npz(folder/'matrix.npz').tocsr()
    with np.load(folder/'vectors.npz',allow_pickle=False) as archive:
        b, reference = archive['rhs'], archive['reference']
    backend = settings['backend']
    if backend not in ('scipy','mumps','superlu_dist'):
        raise ValueError('unknown linear backend')
    if backend != 'scipy':
        from petsc4py import PETSc
        if not PETSc.Sys.hasExternalPackage(backend):
            raise RuntimeError('PETSc lacks '+backend)
        if np.dtype(PETSc.ScalarType) != np.dtype(np.float64):
            raise RuntimeError('requires real float64 PETSc')
    baseline_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    cpu_start, start = process_time(), perf_counter()
    if backend == 'scipy':
        x = spsolve(a,b)
        setup_seconds = None
    else:
        matrix = PETSc.Mat().createAIJ(size=a.shape,csr=(
            a.indptr.astype(PETSc.IntType),a.indices.astype(PETSc.IntType),a.data),comm=PETSc.COMM_SELF)
        matrix.assemble()
        rhs = PETSc.Vec().createWithArray(b.copy(),comm=PETSc.COMM_SELF)
        answer = rhs.duplicate()
        solver = PETSc.KSP().create(PETSc.COMM_SELF)
        solver.setType('preonly')
        solver.getPC().setType('lu')
        solver.getPC().setFactorSolverType(backend)
        solver.setOperators(matrix)
        solver.setErrorIfNotConverged(True)
        setup_seconds = perf_counter()-start
        solver.solve(rhs,answer)
        if solver.getConvergedReason() <= 0:
            raise RuntimeError('PETSc linear solve did not converge')
        x = answer.getArray(readonly=True).copy()
    seconds, cpu_seconds = perf_counter()-start, process_time()-cpu_start
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    residual = float(np.linalg.norm(a @ x-b)/max(np.linalg.norm(b),1.))
    difference = float(np.max(np.abs(x-reference)))
    if backend != 'scipy':
        solver.destroy(); answer.destroy(); rhs.destroy(); matrix.destroy()
    return dict(backend=backend,seconds=seconds,cpu_seconds=cpu_seconds,
        setup_seconds=setup_seconds,peak_rss_bytes=peak,baseline_peak_rss_bytes=baseline_rss,
        relative_residual=residual,max_absolute_reference_difference=difference,
        passed=bool(np.isfinite(x).all() and residual<=1e-10 and difference<=1e-9),
        requested_threads={name:os.environ.get(name) for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},
        affinity=sorted(os.sched_getaffinity(0)),shape=list(a.shape),nnz=a.nnz)


if __name__ == '__main__':
    print(json.dumps(trial(json.load(sys.stdin)),allow_nan=False))
