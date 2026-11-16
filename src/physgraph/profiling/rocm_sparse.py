"""Opt-in vendor-library GPU probe. This is SpMV, not a flow linear solver.

Uses the documented HIP runtime and rocSPARSE dcsrmv C APIs via ctypes;
no generated/custom GPU kernel, torch dependency, or change to CPU imports.
"""
import ctypes as ct
from pathlib import Path
from time import perf_counter
import numpy as np


def _bind(library, name, args):
    function = getattr(library, name)
    function.argtypes, function.restype = args, ct.c_int
    def checked(*values):
        status = function(*values)
        if status != 0:
            raise RuntimeError(f'{name} returned status {status}')
    return checked


def probe_spmv(matrix, vector, *, repeats=30, device=0):
    import rocm_sdk
    import _rocm_sdk_libraries
    if matrix.dtype != np.float64 or vector.dtype != np.float64:
        raise ValueError('GPU probe requires float64')
    if type(repeats) is not int or repeats < 3:
        raise ValueError('at least three repeats required')
    a = matrix.tocsr()
    a.check_format(full_check=True)
    if max(*a.shape, a.nnz) > np.iinfo(np.int32).max:
        raise ValueError('probe uses the rocSPARSE int32 ABI')
    if vector.shape != (a.shape[1],) or not np.isfinite(a.data).all() or not np.isfinite(vector).all():
        raise ValueError('invalid probe matrix/vector')
    rocm_sdk.preload_libraries('amd_comgr', 'amdhip64', 'hiprtc', 'rocm_kpack')
    hip = ct.CDLL(str(rocm_sdk.find_libraries('amdhip64')[0]))
    sparse_path = Path(_rocm_sdk_libraries.__file__).parent/'lib/librocsparse.so.1'
    sparse = ct.CDLL(str(sparse_path))
    ptr, integer, size = ct.c_void_p, ct.c_int, ct.c_size_t
    init = _bind(hip, 'hipInit', [ct.c_uint])
    select = _bind(hip, 'hipSetDevice', [integer])
    name = _bind(hip, 'hipDeviceGetName', [ptr, integer, integer])
    pci = _bind(hip, 'hipDeviceGetPCIBusId', [ptr, integer, integer])
    memory = _bind(hip, 'hipMemGetInfo', [ct.POINTER(size), ct.POINTER(size)])
    malloc = _bind(hip, 'hipMalloc', [ct.POINTER(ptr), size])
    free = _bind(hip, 'hipFree', [ptr])
    copy = _bind(hip, 'hipMemcpy', [ptr, ptr, size, integer])
    sync = _bind(hip, 'hipDeviceSynchronize', [])
    create = _bind(sparse, 'rocsparse_create_handle', [ct.POINTER(ptr)])
    destroy = _bind(sparse, 'rocsparse_destroy_handle', [ptr])
    create_descr = _bind(sparse, 'rocsparse_create_mat_descr', [ct.POINTER(ptr)])
    destroy_descr = _bind(sparse, 'rocsparse_destroy_mat_descr', [ptr])
    mv = _bind(sparse, 'rocsparse_dcsrmv', [ptr, integer, integer, integer, integer,
               ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr])
    start = perf_counter()
    init(0); select(device)
    gpu_name, bus = ct.create_string_buffer(256), ct.create_string_buffer(256)
    name(gpu_name, 256, device); pci(bus, 256, device)
    available, total = size(), size()
    observations = []
    def sample_memory():
        memory(ct.byref(available), ct.byref(total))
        observations.append(total.value-available.value)
    sample_memory()
    handle, descr, allocations = ptr(), ptr(), []
    host = [np.ascontiguousarray(a.data), np.asarray(a.indptr,dtype=np.int32),
            np.asarray(a.indices,dtype=np.int32), np.ascontiguousarray(vector),
            np.zeros(a.shape[0], dtype=np.float64)]
    alpha, beta = ct.c_double(1.), ct.c_double(0.)
    try:
        create(ct.byref(handle)); create_descr(ct.byref(descr))
        for array in host:
            allocation = ptr()
            malloc(ct.byref(allocation), array.nbytes)
            allocations.append(allocation)
            sample_memory()
        def upload():
            for array, allocation in zip(host, allocations):
                copy(allocation, array.ctypes.data, array.nbytes, 1)
        def compute():
            data, row, col, x, y = allocations
            mv(handle, 111, *a.shape, a.nnz, ct.byref(alpha), descr,
               data, row, col, None, x, ct.byref(beta), y)
            sync()
        def download():
            copy(host[-1].ctypes.data, allocations[-1], host[-1].nbytes, 2)
        upload(); compute(); download(); sample_memory()
        cold = perf_counter()-start
        reference = a @ vector
        error = float(np.max(np.abs(host[-1]-reference)))
        residual = float(np.linalg.norm(host[-1]-reference)/max(1.,np.linalg.norm(reference)))
        resident, transferred, cpu = [], [], []
        for _ in range(repeats):
            tick = perf_counter(); compute(); resident.append(perf_counter()-tick)
            tick = perf_counter(); upload(); compute(); download(); transferred.append(perf_counter()-tick)
            error = max(error, float(np.max(np.abs(host[-1]-reference))))
            residual = max(residual, float(np.linalg.norm(host[-1]-reference)/max(1.,np.linalg.norm(reference))))
            tick = perf_counter(); a @ vector; cpu.append(perf_counter()-tick)
            sample_memory()
        return dict(device_name=gpu_name.value.decode(), pci_bus=bus.value.decode(),
            reported_vram_bytes=total.value, precision='float64', shape=list(a.shape), nnz=a.nnz,
            passed=bool(np.isfinite(host[-1]).all() and error<=1e-10 and residual<=1e-10),
            max_absolute_error=error, relative_error=residual, cold_setup_compute_seconds=cold,
            gpu_resident_median_seconds=float(np.median(resident)),
            gpu_transfer_inclusive_median_seconds=float(np.median(transferred)),
            cpu_spmv_median_seconds=float(np.median(cpu)), repeats=repeats,
            resident_seconds=resident, transfer_inclusive_seconds=transferred, cpu_seconds=cpu,
            explicit_device_allocation_peak_bytes=sum(array.nbytes for array in host),
            sampled_device_used_peak_bytes=max(observations), initial_device_used_bytes=observations[0],
            memory_method='hipMemGetInfo at allocation and synchronized iteration boundaries; device-wide sampled use includes other processes and may miss library transient peaks; explicit allocation peak excludes library/context allocations',
            timing='perf_counter; sync every GPU call; warm median excludes allocation/context, transfer-inclusive includes all CSR/vector uploads and result download; cold includes context, allocation, first upload/compute/download',
            operation='vendor rocSPARSE dcsrmv without analysis, int32 CSR; no linear solve',
            libraries=dict(hip=str(rocm_sdk.find_libraries('amdhip64')[0]),rocsparse=str(sparse_path)))
    finally:
        for allocation in reversed(allocations):
            free(allocation)
        if descr.value:
            destroy_descr(descr)
        if handle.value:
            destroy(handle)
