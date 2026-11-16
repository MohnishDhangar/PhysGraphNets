"""Real-device refactorization checks plus CPU factor-structure validation."""
import os
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from physgraph.fem.rocm_refactor import bundle_factors, RocmRefactorSolver


def test_factor_bundle_preserves_structural_zero():
    lower = csr_matrix(([1.,0.,1.], [0,0,1], [0,1,3]),shape=(2,2))
    upper = csr_matrix([[2.,3.],[0.,4.]])
    bundled = bundle_factors(lower,upper)
    assert bundled.nnz == 4
    np.testing.assert_array_equal(bundled.toarray(),[[2.,3.],[0.,4.]])


@pytest.mark.skipif(os.environ.get('PHYSGRAPH_GPU_TEST')!='1',reason='explicit real GPU test run required')
def test_refactor_changed_values_rhs_and_pattern():
    from physgraph.fem.rocm_flow_solver import rocm_identity
    rocm_identity()
    # Nonsymmetric saddle-point matrix requiring pivot permutations.
    a = csr_matrix([[.01,2.,1.,0.],[3.,4.,0.,1.],[1.,0.,0.,0.],[0.,1.,0.,0.]])
    solver = RocmRefactorSolver(a)
    try:
        for scale in (1.,1.01,.99):
            changed = a.copy(); changed.data *= 1+(.1*scale)*np.sin(np.arange(a.nnz))
            rhs = np.arange(1.,5.)*scale
            np.testing.assert_allclose(solver(changed,rhs),np.linalg.solve(changed.toarray(),rhs),rtol=1e-12,atol=1e-12)
        with pytest.raises(ValueError,match='sparsity pattern'):
            solver(csr_matrix(np.eye(4)),np.ones(4))
        assert len(solver.records)==3
    finally:
        solver.close()
