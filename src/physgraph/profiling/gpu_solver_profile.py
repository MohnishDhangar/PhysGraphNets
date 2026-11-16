"""Synchronized operation profiling of the original GPU solver candidate."""
from pathlib import Path
from time import perf_counter
from physgraph.experiments.gpu_solver_checks import run_gpu_solver_checks
from physgraph.fem import rocm_algebra


def profile_gpu_solver(*,fixtures,output):
    original=rocm_algebra.RocmAlgebra
    class ProfiledAlgebra(original):
        def __init__(self,*args,**kwargs):
            self.operation_times={}
            super().__init__(*args,**kwargs)
        def memory_report(self):
            report=super().memory_report()
            return dict(**report,synchronized_operation_profile=self.operation_times,
                profile_note='instrumented diagnostic run, not performance acceptance; CSR includes upload/analysis, so nested timings must not be added')
    def measured(name):
        method=getattr(original,name)
        def call(self,*args,**kwargs):
            self.sync()
            start=perf_counter()
            answer=method(self,*args,**kwargs)
            self.sync()
            row=self.operation_times.setdefault(name,dict(calls=0,seconds=0.))
            row['calls']+=1;row['seconds']+=perf_counter()-start
            return answer
        return call
    for name in ('csr','upload','precondition','matvec','dot','norm','axpy','scale','assign'):
        setattr(ProfiledAlgebra,name,measured(name))
    rocm_algebra.RocmAlgebra=ProfiledAlgebra
    try:
        return run_gpu_solver_checks(fixtures=fixtures,output=Path(output))
    finally:
        rocm_algebra.RocmAlgebra=original
