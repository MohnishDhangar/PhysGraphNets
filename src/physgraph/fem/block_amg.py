"""Fixed-cycle velocity/pressure block preconditioning; optional PyAMG setup.

For [F G; B 0], approximate the positive Schur operator B diag(F)^-1 G.
All application operations are SpMV/diagonal operations, suitable for GPU reuse.
"""
import os
from pathlib import Path
import sys
import numpy as np
from scipy.sparse import csr_matrix,diags
from scipy.sparse.linalg import splu


def load_pyamg():
    site=os.environ.get('PHYSGRAPH_AMG_SITE')
    if site and str(Path(site).resolve()) not in sys.path:
        sys.path.insert(0,str(Path(site).resolve()))
    import pyamg
    return pyamg


def hierarchy(matrix):
    pyamg=load_pyamg()
    # PyAMG's setup spectral estimates use NumPy's legacy RNG. Isolate the seed.
    state=np.random.get_state()
    try:
        np.random.seed(713)
        mg=pyamg.smoothed_aggregation_solver(matrix,symmetry='symmetric',
            max_coarse=32,max_levels=12,coarse_solver='splu',
            presmoother=('jacobi',{'iterations':2}),postsmoother=('jacobi',{'iterations':2}))
    finally:
        np.random.set_state(state)
    levels=[]
    for level in mg.levels:
        a=csr_matrix(level.A)
        if not hasattr(level,'P'):
            inverse=np.linalg.inv(a.toarray())
            if not np.isfinite(inverse).all():raise RuntimeError('invalid AMG coarse inverse')
            levels.append(dict(A=a,inverse=csr_matrix(inverse)))
        else:
            diagonal=a.diagonal()
            if np.any(diagonal<=0):raise RuntimeError('AMG requires positive smoother diagonal')
            bound=np.max(np.asarray(abs(a).sum(axis=1)).ravel()/diagonal)
            omega=1.4/bound
            levels.append(dict(A=a,P=csr_matrix(level.P),R=csr_matrix(level.R),
                               D=diags(omega/diagonal,format='csr')))
    return levels


def vcycle(levels,b,index=0):
    level=levels[index]
    if 'inverse' in level:return level['inverse']@b
    x=np.zeros_like(b)
    for _ in range(3):x+=level['D']@(b-level['A']@x)
    x+=level['P']@vcycle(levels,level['R']@(b-level['A']@x),index+1)
    for _ in range(3):x+=level['D']@(b-level['A']@x)
    return x


class BlockAMG:
    def __init__(self,a):
        a=a.tocsr()
        zero=np.flatnonzero(a.diagonal()==0.)
        if not len(zero):raise ValueError('expected pressure saddle block')
        self.nv=int(zero[0]);self.n=a.shape[0]
        if np.any(a.diagonal()[:self.nv]<=0) or a[self.nv:,self.nv:].nnz:
            raise ValueError('block AMG currently supports traction pressure without mean-gauge augmentation')
        self.F=a[:self.nv,:self.nv].tocsr()
        self.G=a[:self.nv,self.nv:].tocsr()
        self.B=a[self.nv:,:self.nv].tocsr()
        self.S=(self.B@diags(1/self.F.diagonal())@self.G).tocsr()
        self.velocity=hierarchy(((self.F+self.F.T)*.5).tocsr())
        self.pressure_factor=splu(self.S.tocsc())
    def apply(self,r):
        u=vcycle(self.velocity,r[:self.nv])
        p=self.pressure_factor.solve(self.B@u-r[self.nv:])
        u-=vcycle(self.velocity,self.G@p)
        return np.r_[u,p]
