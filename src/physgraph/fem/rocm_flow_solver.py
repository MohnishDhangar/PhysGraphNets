"""Experimental GPU iteration backend with explicitly timed CPU ILUT setup."""
from importlib.metadata import version
import os
from pathlib import Path
import sys
from time import perf_counter
import numpy as np
from scipy.sparse.linalg import spilu
from physgraph.fem.resident_gmres import resident_gmres


SETTINGS = dict(rtol=1e-12,restart=30,max_iterations=120,timeout_seconds=60.,
                drop_tol=1e-6,fill_factor=20,preconditioner_reuse='within timestep only')


def rocm_identity():
    site = os.environ.get('PHYSGRAPH_ROCM_SITE')
    if site and str(Path(site).resolve()) not in sys.path:
        sys.path.insert(0,str(Path(site).resolve()))
    try:
        import rocm_sdk
    except ImportError as exc:
        raise RuntimeError('experimental rocm_gmres requires isolated ROCm SDK; set PHYSGRAPH_ROCM_SITE') from exc
    return dict(name='rocm_gmres',settings=dict(SETTINGS),precision='float64',
        sdk=str(Path(rocm_sdk.__file__).resolve()),packages={name:version(name) for name in
            ('rocm','rocm-sdk-core','rocm-sdk-libraries')},device_index=0,
        visibility={name:os.environ.get(name) for name in ('HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES')},
        requested_threads={name:os.environ.get(name) for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')})


class RocmFlowSolver:
    def __init__(self):
        self.records = []
        self.factor = None
        self.step = 0

    def begin_step(self, step):
        # Reset at every timestep so restart does not need hidden preconditioner state.
        self.factor = None
        self.step = step

    def __call__(self, matrix, rhs):
        from physgraph.fem.rocm_algebra import RocmAlgebra
        start = perf_counter()
        a, b = matrix.tocsr(), np.asarray(rhs)
        if a.dtype != np.float64 or b.dtype != np.float64 or a.shape != (len(b),len(b)):
            raise ValueError('rocm_gmres requires a real float64 square system')
        if not np.isfinite(a.data).all() or not np.isfinite(b).all():
            raise ValueError('rocm_gmres requires finite data')
        if not np.any(b):
            self.records.append(dict(step=self.step,iterations=0,relative_residual=0.,seconds=perf_counter()-start,zero_rhs=True))
            return np.zeros_like(b)
        setup = perf_counter()
        reused = self.factor is not None
        if self.factor is None:
            lu = spilu(a.tocsc(),drop_tol=SETTINGS['drop_tol'],fill_factor=SETTINGS['fill_factor'])
            self.factor = (np.argsort(lu.perm_r),np.argsort(lu.perm_c),lu.L.tocsr(),lu.U.tocsr())
        rows,cols,lower,upper = self.factor
        cpu_setup = perf_counter()-setup
        transformed = a[rows][:,cols].tocsr()
        gpu_start = perf_counter()
        with RocmAlgebra(transformed,lower,upper) as algebra:
            device_rhs = algebra.upload(b[rows])
            device_solution, info = resident_gmres(algebra,device_rhs,
                **{key:SETTINGS[key] for key in ('rtol','restart','max_iterations','timeout_seconds')})
            x = np.empty_like(b); x[cols] = algebra.download(device_solution)
            memory = algebra.memory_report()
        gpu_seconds = perf_counter()-gpu_start
        residual = float(np.linalg.norm(a@x-b)/max(np.linalg.norm(b),1.))
        if not np.isfinite(x).all() or not np.isfinite(residual) or residual>1e-10:
            raise RuntimeError('GPU original-system residual exceeds 1e-10')
        self.records.append(dict(step=self.step,**info,original_scaled_residual=residual,
            cpu_preconditioner_seconds=cpu_setup,preconditioner_reused=reused,
            gpu_setup_transfer_iteration_cleanup_seconds=gpu_seconds,seconds=perf_counter()-start,**memory))
        return x

    def close(self):
        self.factor = None
