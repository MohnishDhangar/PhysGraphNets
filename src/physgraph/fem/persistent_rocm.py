"""Retain factors, sparse analysis and workspaces across current-matrix updates."""
import numpy as np
from physgraph.fem.rocm_algebra import RocmAlgebra


class PersistentRocmAlgebra(RocmAlgebra):
    def csr(self,matrix,triangular=None):
        if triangular is not None:return super().csr(matrix,triangular)
        a=matrix.tocsr(copy=True);a.sum_duplicates();a.sort_indices()
        self.capacity=max(1,2*a.nnz)
        self.pattern=(a.indptr.copy(),a.indices.copy())
        descr=self.object(self.sparse,'rocsparse_create_mat_descr','rocsparse_destroy_mat_descr')
        data=self.buffer(self.capacity*8);col=self.buffer(self.capacity*4)
        row=self.buffer((self.n+1)*4)
        self._copy_csr(a,data,row,col,True)
        return a.nnz,descr,data,row,col
    def _copy_csr(self,a,data,row,col,pattern):
        values=np.ascontiguousarray(a.data)
        self.memcpy(data,values.ctypes.data,values.nbytes,1)
        if pattern:
            rows=np.asarray(a.indptr,dtype=np.int32);cols=np.asarray(a.indices,dtype=np.int32)
            self.memcpy(row,rows.ctypes.data,rows.nbytes,1)
            self.memcpy(col,cols.ctypes.data,cols.nbytes,1)
    def update_matrix(self,matrix):
        a=matrix.tocsr(copy=True);a.sum_duplicates();a.sort_indices()
        if a.shape!=(self.n,self.n) or a.dtype!=np.float64 or a.nnz>np.iinfo(np.int32).max:
            raise ValueError('incompatible current matrix')
        if a.nnz>self.capacity:
            # Bounded growth for changed sparsity; old allocation is freed on refresh.
            self.a=self.csr(a)
        else:
            _,descr,data,row,col=self.a
            pattern=not (np.array_equal(a.indptr,self.pattern[0]) and np.array_equal(a.indices,self.pattern[1]))
            self._copy_csr(a,data,row,col,pattern)
            if pattern:self.pattern=(a.indptr.copy(),a.indices.copy())
            self.a=a.nnz,descr,data,row,col
