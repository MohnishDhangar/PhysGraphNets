"""Experimental vendor sparse LU refactorization; not a trajectory backend.

CPU SuperLU chooses the initial sparsity and pivots. Subsequent numerical LU
factorization and triangular solution run on GPU. Pattern changes are rejected;
fixed pivots are accepted only when the original-system residual passes.
API: https://rocmdocs.amd.com/projects/rocSOLVER/en/latest/reference/refact.html
"""
import ctypes as ct
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu
from physgraph.fem.rocm_algebra import RocmAlgebra, checked


def bundle_factors(lower, upper):
    """Preserve stored zero fill entries when bundling (L-I)+U."""
    lower, upper = lower.tocoo(), upper.tocoo()
    keep = lower.row != lower.col
    return coo_matrix((np.concatenate((lower.data[keep], upper.data)),
        (np.concatenate((lower.row[keep], upper.row)),
         np.concatenate((lower.col[keep], upper.col)))), shape=lower.shape).tocsr()


class RocmRefactorSolver:
    def __init__(self, matrix):
        start = perf_counter()
        a = self.validate(matrix)
        lu = splu(a.tocsc())
        self.rows, self.cols = np.argsort(lu.perm_r), np.argsort(lu.perm_c)
        self.pattern = a[self.rows][:, self.cols].tocsr()
        self.pattern.sort_indices()
        factors = bundle_factors(lu.L, lu.U)
        self.cpu_setup_seconds = perf_counter()-start
        self.records = []
        self.gpu = None
        tick = perf_counter()
        try:
            self.gpu = g = RocmAlgebra(self.pattern, None, None)
            import _rocm_sdk_libraries
            library = ct.CDLL(str(Path(_rocm_sdk_libraries.__file__).parent/'lib/librocsolver.so.0'))
            p, i = ct.c_void_p, ct.c_int
            info = p()
            checked(library, 'rocsolver_create_rfinfo', [ct.POINTER(p), p])(ct.byref(info), g.bh)
            release = checked(library, 'rocsolver_destroy_rfinfo', [p])
            g.cleanup.append(lambda: release(info))
            self.info = info
            self.analysis = checked(library, 'rocsolver_dcsrrf_analysis',
                [p,i,i,i,p,p,p,i,p,p,p,p,p,p,i,p])
            self.refactor = checked(library, 'rocsolver_dcsrrf_refactlu',
                [p,i,i,p,p,p,i,p,p,p,p,p,p])
            self.solve = checked(library, 'rocsolver_dcsrrf_solve',
                [p,i,i,i,p,p,p,p,p,p,i,p])
            self.factor = g.csr(factors)
            self.pivots = g.upload(np.arange(g.n, dtype=np.int32))
            self.rhs = g.zeros()
            na, _, av, ar, ac = g.a
            nt, _, tv, tr, tc = self.factor
            self.analysis(g.bh,g.n,1,na,ar,ac,av,nt,tr,tc,tv,
                          self.pivots,self.pivots,self.rhs,g.n,info)
            g.sync(); g.sample()
            self.gpu_setup_seconds = perf_counter()-tick
        except BaseException:
            self.close()
            raise

    @staticmethod
    def validate(matrix):
        a = matrix.tocsr(copy=True)
        a.sum_duplicates(); a.sort_indices(); a.check_format(full_check=True)
        if a.dtype != np.float64 or a.shape[0] != a.shape[1] or a.shape[0] == 0:
            raise ValueError('refactor requires a nonempty square float64 matrix')
        if max(a.shape[0],a.nnz) > np.iinfo(np.int32).max or not np.isfinite(a.data).all():
            raise ValueError('refactor requires finite int32-indexed data')
        return a

    def __call__(self, matrix, rhs):
        start = perf_counter()
        a, b = self.validate(matrix), np.asarray(rhs)
        if b.shape != (a.shape[0],) or b.dtype != np.float64 or not np.isfinite(b).all():
            raise ValueError('refactor requires a finite float64 right-hand side')
        if a.shape != self.pattern.shape:
            raise ValueError('refactor sparsity pattern changed')
        transformed = a[self.rows][:,self.cols].tocsr(); transformed.sort_indices()
        if not (np.array_equal(transformed.indptr,self.pattern.indptr) and
                np.array_equal(transformed.indices,self.pattern.indices)):
            raise ValueError('refactor sparsity pattern changed')
        g = self.gpu
        if g is None: raise RuntimeError('refactor solver is closed')
        na, _, av, ar, ac = g.a
        nt, _, tv, tr, tc = self.factor
        values = np.ascontiguousarray(transformed.data)
        vector = np.ascontiguousarray(b[self.rows])
        g.memcpy(av,values.ctypes.data,values.nbytes,1)
        g.memcpy(self.rhs,vector.ctypes.data,vector.nbytes,1)
        g.sync()
        upload_seconds = perf_counter()-start
        tick = perf_counter()
        self.refactor(g.bh,g.n,na,ar,ac,av,nt,tr,tc,tv,self.pivots,self.pivots,self.info)
        g.sync(); g.sample()
        factor_seconds = perf_counter()-tick
        tick = perf_counter()
        self.solve(g.bh,g.n,1,nt,tr,tc,tv,self.pivots,self.pivots,self.rhs,g.n,self.info)
        g.sync()
        x = np.empty_like(b); x[self.cols] = g.download(self.rhs)
        solve_seconds = perf_counter()-tick
        tick = perf_counter()
        norm_b = np.linalg.norm(b)
        residual = b-a@x
        history = []
        # Fixed pivots may lose accuracy. Correct using the GPU factors, never
        # a CPU solution; retain the same original-system acceptance thresholds.
        for refinement in range(4):
            norm_error = np.linalg.norm(residual)
            scaled = float(norm_error/max(1.,norm_b))
            relative = float(norm_error/norm_b) if norm_b else float(norm_error)
            history.append(relative if np.isfinite(relative) else None)
            if not np.isfinite(relative) or (scaled <= 1e-10 and relative <= 2e-12) or refinement == 3:
                break
            vector = np.ascontiguousarray(residual[self.rows])
            g.memcpy(self.rhs,vector.ctypes.data,vector.nbytes,1)
            self.solve(g.bh,g.n,1,nt,tr,tc,tv,self.pivots,self.pivots,self.rhs,g.n,self.info)
            g.sync()
            x[self.cols] += g.download(self.rhs)
            residual = b-a@x
        passed = bool(np.isfinite(x).all() and scaled <= 1e-10 and relative <= 2e-12)
        self.records.append(dict(passed=passed,original_scaled_residual=scaled if np.isfinite(scaled) else None,
            original_relative_residual=relative if np.isfinite(relative) else None,
            refinement_residual_history=history,refinement_steps=refinement,
            residual_check_and_refinement_seconds=perf_counter()-tick,upload_permutation_seconds=upload_seconds,
            gpu_refactor_seconds=factor_seconds,gpu_solve_download_seconds=solve_seconds,
            seconds=perf_counter()-start))
        if not passed: raise RuntimeError(f'GPU refactor residual failed: {relative}')
        return x

    def close(self):
        if self.gpu is not None:
            self.gpu.close()
            self.gpu = None
