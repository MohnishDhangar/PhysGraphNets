"""Verify the shared Krylov algorithm without a GPU or optional dependencies."""
import numpy as np
import pytest
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spilu, spsolve_triangular
from physgraph.fem.resident_gmres import resident_gmres


class CpuAlgebra:
    def __init__(self, a, preconditioner=None):
        self.a, self.n = a,len(a)
        self.preconditioner = preconditioner
    def zeros(self): return np.zeros(self.n)
    def zero(self,x): x[:] = 0
    def copy(self,x): return x.copy()
    def assign(self,x,y): x[:] = y
    def scale(self,x,v): x *= v
    def axpy(self,x,y,v): x += v*y
    def dot(self,x,y): return float(x@y)
    def norm(self,x): return float(np.linalg.norm(x))
    def matvec(self,x,y): y[:] = self.a@x
    def precondition(self,x,y): y[:] = x if self.preconditioner is None else self.preconditioner(x)


@pytest.mark.parametrize('restart',[3,10])
def test_right_preconditioned_gmres_true_residual(restart):
    rng = np.random.default_rng(14)
    a = rng.normal(size=(10,10)) + 8*np.eye(10)
    exact = rng.normal(size=10); b = a@exact
    result, info = resident_gmres(CpuAlgebra(a,lambda x:x/np.diag(a)),b,restart=restart)
    np.testing.assert_allclose(result,exact,atol=1e-10,rtol=0)
    assert np.linalg.norm(a@result-b)/np.linalg.norm(b)<=1e-12
    assert info['iterations']>0


def test_indefinite_saddle_system_and_ilu_permutations():
    rng=np.random.default_rng(7)
    k=rng.normal(size=(8,8)); k=k.T@k+np.eye(8)
    b=rng.normal(size=(3,8))
    a=np.block([[k,b.T],[b,np.zeros((3,3))]])
    rhs=rng.normal(size=11)
    lu=spilu(csc_matrix(a),drop_tol=1e-6,fill_factor=20)
    rows,cols=np.argsort(lu.perm_r),np.argsort(lu.perm_c)
    def precondition(x):
        return spsolve_triangular(lu.U.tocsr(),spsolve_triangular(lu.L.tocsr(),x,lower=True),lower=False)
    algebra=CpuAlgebra(a[rows][:,cols],precondition)
    transformed,info=resident_gmres(algebra,rhs[rows])
    answer=np.empty(11);answer[cols]=transformed
    np.testing.assert_allclose(answer,np.linalg.solve(a,rhs),atol=1e-10,rtol=0)


def test_zero_rhs_and_nonconvergence_are_explicit():
    result,info=resident_gmres(CpuAlgebra(np.eye(2)),np.zeros(2))
    assert np.array_equal(result,np.zeros(2)) and info['iterations']==0
    with pytest.raises(RuntimeError,match='failed'):
        resident_gmres(CpuAlgebra(np.zeros((2,2))),np.ones(2),max_iterations=2)
    with pytest.raises(ValueError,match='limits'):
        resident_gmres(CpuAlgebra(np.eye(2)),np.ones(2),rtol=1e-5)
