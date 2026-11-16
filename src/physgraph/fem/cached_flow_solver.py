"""Persistent right-preconditioned CPU/GPU solvers with checked current matrices.

CPU sparse LU supplies a reusable preconditioner, never the current solution.
Refresh at global 32-step boundaries or after slow/nonconvergent iterations.
"""
from time import perf_counter
import numpy as np
from scipy.sparse.linalg import splu
from physgraph.fem.resident_gmres import resident_gmres

SETTINGS=dict(rtol=1e-12,restart=30,max_iterations=120,timeout_seconds=60.,
              refresh_steps=32,refresh_iterations=12,preconditioner='CPU SuperLU exact LU reused as right preconditioner')


class CpuCachedAlgebra:
    def __init__(self,a,lu,rows,cols):
        self.a,self.lu,self.rows,self.cols=a,lu,rows,cols
        self.n=a.shape[0]
    def zeros(self):return np.zeros(self.n)
    def zero(self,x):x.fill(0)
    def copy(self,x):return x.copy()
    def assign(self,x,y):x[:]=y
    def scale(self,x,v):x*=v
    def axpy(self,x,y,v):x+=v*y
    def dot(self,x,y):return float(x@y)
    def norm(self,x):return float(np.linalg.norm(x))
    def matvec(self,x,y):y[:]=self.a@x
    def precondition(self,x,y):
        b=np.empty(self.n);b[self.rows]=x
        y[:]=self.lu.solve(b)[self.cols]
    def update_matrix(self,a):self.a=a
    def close(self):pass


class CachedFlowSolver:
    def __init__(self,device):
        if device not in ('cpu','gpu'):raise ValueError('device must be cpu or gpu')
        self.device=device;self.records=[];self.step=0;self.block=None
        self.algebra=None;self.workspace={};self.rows=self.cols=None
        self.refresh=False;self.setup_count=0
    def begin_step(self,step):
        block=(step-1)//SETTINGS['refresh_steps']
        if self.block!=block:
            self.close();self.block=block
        self.step=step
    def close(self):
        if self.algebra is not None:self.algebra.close()
        self.algebra=None;self.workspace={};self.rows=self.cols=None
        self.refresh=False
    def build(self,a):
        self.close()
        lu=splu(a.tocsc())
        self.rows,self.cols=np.argsort(lu.perm_r),np.argsort(lu.perm_c)
        transformed=a[self.rows][:,self.cols].tocsr()
        if self.device=='gpu':
            from physgraph.fem.persistent_rocm import PersistentRocmAlgebra
            self.algebra=PersistentRocmAlgebra(transformed,lu.L.tocsr(),lu.U.tocsr())
            self.rhs=self.algebra.zeros()
        else:self.algebra=CpuCachedAlgebra(transformed,lu,self.rows,self.cols)
        self.setup_count+=1
    def __call__(self,matrix,rhs):
        start=perf_counter();a=matrix.tocsr();b=np.asarray(rhs)
        if b.ndim!=1 or a.shape!=(len(b),len(b)) or a.dtype!=np.float64 or b.dtype!=np.float64:
            raise ValueError('cached solver requires float64 square matrix and vector')
        if not np.isfinite(a.data).all() or not np.isfinite(b).all():raise ValueError('nonfinite system')
        if not np.any(b):
            self.records.append(dict(step=self.step,zero_rhs=True,iterations=0,seconds=perf_counter()-start))
            return np.zeros_like(b)
        setup_seconds=update_seconds=0.;failures=[]
        rebuilt=self.algebra is None or self.refresh
        for attempt in range(2):
            tick=perf_counter()
            if self.algebra is None or self.refresh or attempt:
                self.build(a);setup_seconds+=perf_counter()-tick;rebuilt=True
            else:
                self.algebra.update_matrix(a[self.rows][:,self.cols].tocsr())
                update_seconds+=perf_counter()-tick
            tick=perf_counter()
            transformed_rhs=np.ascontiguousarray(b[self.rows])
            if self.device=='gpu':
                self.algebra.memcpy(self.rhs,transformed_rhs.ctypes.data,b.nbytes,1)
                vector=self.rhs
            else:vector=transformed_rhs
            try:
                solution,info=resident_gmres(self.algebra,vector,workspace=self.workspace,
                    **{k:SETTINGS[k] for k in ('rtol','restart','max_iterations','timeout_seconds')})
                value=self.algebra.download(solution) if self.device=='gpu' else solution
                x=np.empty_like(b);x[self.cols]=value
                scaled=float(np.linalg.norm(a@x-b)/max(1.,np.linalg.norm(b)))
                relative=float(np.linalg.norm(a@x-b)/np.linalg.norm(b))
                if not np.isfinite(x).all() or scaled>1e-10 or relative>2e-12:
                    raise RuntimeError(f'current-system residual failed: {relative}')
                break
            except RuntimeError as exc:
                failures.append(str(exc))
                if attempt:raise
        iteration_seconds=perf_counter()-tick
        self.refresh=info['iterations']>SETTINGS['refresh_iterations']
        memory=self.algebra.memory_report() if self.device=='gpu' else {}
        self.records.append(dict(step=self.step,**info,original_scaled_residual=scaled,
            original_relative_residual=relative,setup_seconds=setup_seconds,matrix_update_seconds=update_seconds,
            iteration_seconds=iteration_seconds,preconditioner_rebuilt=rebuilt,setup_count=self.setup_count,
            retry_failures=failures,seconds=perf_counter()-start,**memory))
        return x
