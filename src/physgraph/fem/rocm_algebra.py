"""Resident float64 vectors and sparse triangular solves using vendor ROCm APIs.

Only imported by the opt-in experimental solver. No custom device kernels.
HIP/rocSPARSE/rocBLAS use their default stream and host scalar pointer mode.
"""
import ctypes as ct
from pathlib import Path
import numpy as np


def checked(library, name, types):
    function = getattr(library,name)
    function.argtypes, function.restype = types, ct.c_int
    def call(*args):
        status = function(*args)
        if status:
            raise RuntimeError(f'{name} returned status {status}')
    return call


class RocmAlgebra:
    def __init__(self, matrix, lower, upper, device=0):
        import rocm_sdk
        import _rocm_sdk_libraries
        rocm_sdk.preload_libraries('amd_comgr','amdhip64','hiprtc','rocm_kpack')
        self.cleanup = []
        self.allocated = self.peak_allocated = 0
        self.samples = []
        self.n = matrix.shape[0]
        p, i, z = ct.c_void_p, ct.c_int, ct.c_size_t
        lib = Path(_rocm_sdk_libraries.__file__).parent/'lib'
        self.hip = ct.CDLL(str(rocm_sdk.find_libraries('amdhip64')[0]))
        self.sparse = ct.CDLL(str(lib/'librocsparse.so.1'))
        self.blas = ct.CDLL(str(lib/'librocblas.so.5'))
        self.malloc = checked(self.hip,'hipMalloc',[ct.POINTER(p),z])
        self.free = checked(self.hip,'hipFree',[p])
        self.memcpy = checked(self.hip,'hipMemcpy',[p,p,z,i])
        self.memset = checked(self.hip,'hipMemset',[p,i,z])
        self.sync = checked(self.hip,'hipDeviceSynchronize',[])
        self.meminfo = checked(self.hip,'hipMemGetInfo',[ct.POINTER(z),ct.POINTER(z)])
        self.mv = checked(self.sparse,'rocsparse_dcsrmv',[p,i,i,i,i,p,p,p,p,p,p,p,p,p])
        self.tri_size = checked(self.sparse,'rocsparse_dcsrsv_buffer_size',[p,i,i,i,p,p,p,p,p,ct.POINTER(z)])
        self.tri_analysis = checked(self.sparse,'rocsparse_dcsrsv_analysis',[p,i,i,i,p,p,p,p,p,i,i,p])
        self.tri_solve = checked(self.sparse,'rocsparse_dcsrsv_solve',[p,i,i,i,p,p,p,p,p,p,p,p,i,p])
        self.bcopy = checked(self.blas,'rocblas_dcopy',[p,i,p,i,p,i])
        self.bscale = checked(self.blas,'rocblas_dscal',[p,i,p,p,i])
        self.baxpy = checked(self.blas,'rocblas_daxpy',[p,i,p,p,i,p,i])
        self.bdot = checked(self.blas,'rocblas_ddot',[p,i,p,i,p,i,p])
        self.bnorm = checked(self.blas,'rocblas_dnrm2',[p,i,p,i,p])
        try:
            checked(self.hip,'hipInit',[ct.c_uint])(0)
            checked(self.hip,'hipSetDevice',[i])(device)
            name, pci = ct.create_string_buffer(256), ct.create_string_buffer(256)
            checked(self.hip,'hipDeviceGetName',[p,i,i])(name,256,device)
            checked(self.hip,'hipDeviceGetPCIBusId',[p,i,i])(pci,256,device)
            self.name, self.pci = name.value.decode(), pci.value.decode()
            self.sample()
            self.sh = self.object(self.sparse,'rocsparse_create_handle','rocsparse_destroy_handle')
            self.bh = self.object(self.blas,'rocblas_create_handle','rocblas_destroy_handle')
            # Deterministic BLAS path where supported; still verify actual outputs.
            checked(self.blas,'rocblas_set_atomics_mode',[p,i])(self.bh,0)
            self.a = self.csr(matrix)
            self.lower = None if lower is None else self.csr(lower,'lower')
            self.upper = None if upper is None else self.csr(upper,'upper')
            self.scratch = self.zeros()
            self.sync(); self.sample()
        except BaseException:
            self.close()
            raise

    def object(self, library, create, destroy):
        pointer = ct.c_void_p()
        checked(library,create,[ct.POINTER(ct.c_void_p)])(ct.byref(pointer))
        release = checked(library,destroy,[ct.c_void_p])
        self.cleanup.append(lambda: release(pointer))
        return pointer

    def sample(self):
        available,total = ct.c_size_t(),ct.c_size_t()
        self.meminfo(ct.byref(available),ct.byref(total))
        self.total = total.value
        self.samples.append(total.value-available.value)

    def buffer(self, size):
        pointer = ct.c_void_p()
        self.malloc(ct.byref(pointer),max(1,size))
        self.cleanup.append(lambda: self.free(pointer))
        self.allocated += max(1,size)
        self.peak_allocated = max(self.peak_allocated,self.allocated)
        self.sample()
        return pointer

    def upload(self, value):
        a = np.ascontiguousarray(value)
        pointer = self.buffer(a.nbytes)
        self.memcpy(pointer,a.ctypes.data,a.nbytes,1)
        return pointer

    def csr(self, matrix, triangular=None):
        a = matrix.tocsr(copy=True)
        a.sum_duplicates(); a.sort_indices(); a.check_format(full_check=True)
        if a.dtype != np.float64 or a.shape != (self.n,self.n) or a.nnz > np.iinfo(np.int32).max:
            raise ValueError('ROCm requires square float64 CSR with int32 indices')
        descr = self.object(self.sparse,'rocsparse_create_mat_descr','rocsparse_destroy_mat_descr')
        data,row,col = (self.upload(a.data),self.upload(a.indptr.astype(np.int32)),self.upload(a.indices.astype(np.int32)))
        if triangular is None:
            return a.nnz,descr,data,row,col
        checked(self.sparse,'rocsparse_set_mat_fill_mode',[ct.c_void_p,ct.c_int])(descr,0 if triangular=='lower' else 1)
        # SciPy exports L's explicit unit diagonal. Both matrices use non-unit API.
        info = self.object(self.sparse,'rocsparse_create_mat_info','rocsparse_destroy_mat_info')
        required = ct.c_size_t()
        self.tri_size(self.sh,111,self.n,a.nnz,descr,data,row,col,info,ct.byref(required))
        workspace = self.buffer(required.value)
        self.tri_analysis(self.sh,111,self.n,a.nnz,descr,data,row,col,info,0,0,workspace)
        self.sync(); self.sample()
        return a.nnz,descr,data,row,col,info,workspace

    def zeros(self):
        result = self.buffer(self.n*8)
        self.zero(result)
        return result

    def zero(self, value):
        self.memset(value,0,self.n*8)

    def assign(self, target, source):
        self.bcopy(self.bh,self.n,source,1,target,1)

    def copy(self, source):
        target = self.zeros(); self.assign(target,source)
        return target

    def scale(self, value, coefficient):
        alpha = ct.c_double(coefficient)
        self.bscale(self.bh,self.n,ct.byref(alpha),value,1)

    def axpy(self, target, source, coefficient):
        alpha = ct.c_double(coefficient)
        self.baxpy(self.bh,self.n,ct.byref(alpha),source,1,target,1)

    def dot(self, a, b):
        result = ct.c_double()
        self.bdot(self.bh,self.n,a,1,b,1,ct.byref(result)); self.sync()
        return result.value

    def norm(self, value):
        result = ct.c_double()
        self.bnorm(self.bh,self.n,value,1,ct.byref(result)); self.sync()
        self.sample()
        return result.value

    def matvec(self, value, result):
        nnz,descr,data,row,col = self.a
        one,zero = ct.c_double(1.),ct.c_double(0.)
        self.mv(self.sh,111,self.n,self.n,nnz,ct.byref(one),descr,data,row,col,None,value,ct.byref(zero),result)

    def precondition(self, value, result):
        one = ct.c_double(1.)
        for factor,source,target in ((self.lower,value,self.scratch),(self.upper,self.scratch,result)):
            nnz,descr,data,row,col,info,workspace = factor
            self.tri_solve(self.sh,111,self.n,nnz,ct.byref(one),descr,data,row,col,info,source,target,0,workspace)

    def download(self, value):
        result = np.empty(self.n)
        self.memcpy(result.ctypes.data,value,self.n*8,2); self.sync()
        return result

    def memory_report(self):
        self.sync(); self.sample()
        return dict(device_name=self.name,pci_bus=self.pci,reported_vram_bytes=self.total,
            explicit_device_allocation_peak_bytes=self.peak_allocated,
            sampled_device_used_peak_bytes=max(self.samples),
            memory_method='explicit allocation peak includes CSR/factors/Krylov/triangular buffers; hipMemGetInfo sampled after allocations, setup and reductions is device-wide and may miss transient library peaks')

    def close(self):
        errors = []
        for release in reversed(self.cleanup):
            try:
                release()
            except Exception as exc:
                errors.append(exc)
        self.cleanup.clear()
        if errors:
            raise RuntimeError('ROCm cleanup failed') from errors[0]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
