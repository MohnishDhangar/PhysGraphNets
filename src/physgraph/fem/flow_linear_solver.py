"""Optional CPU sparse backend; SciPy remains the reference and default."""
import os
import numpy as np


def backend_identity(backend):
    if backend == 'scipy':
        return {'name': 'scipy'}
    if backend in ('cpu_cached','rocm_cached'):
        from physgraph.fem.cached_flow_solver import SETTINGS
        identity={}
        if backend=='rocm_cached':
            from physgraph.fem.rocm_flow_solver import rocm_identity
            identity=rocm_identity()
        return {**identity,'name':backend,'settings':dict(SETTINGS)}
    if backend == 'rocm_gmres':
        from physgraph.fem.rocm_flow_solver import rocm_identity
        return rocm_identity()
    if backend != 'mumps':
        raise ValueError('linear_backend must be scipy, mumps or experimental rocm_gmres')
    try:
        from petsc4py import PETSc
    except ImportError as exc:
        raise RuntimeError('mumps requires an optional PETSc/petsc4py installation') from exc
    if not PETSc.Sys.hasExternalPackage('mumps'):
        raise RuntimeError('PETSc was built without MUMPS')
    if np.dtype(PETSc.ScalarType) != np.dtype(np.float64):
        raise RuntimeError('MUMPS backend requires real float64 PETSc')
    return dict(name='mumps', petsc_version=list(PETSc.Sys.getVersion()),
                requested_threads={key: os.environ.get(key) for key in
                    ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')})


def mumps_solve(matrix, rhs):
    """Factor each changing Newton matrix; do not reuse stale numerical factors."""
    from petsc4py import PETSc
    a = matrix.tocsr()
    objects = []
    try:
        mat = PETSc.Mat().createAIJ(size=a.shape, csr=(
            a.indptr.astype(PETSc.IntType), a.indices.astype(PETSc.IntType), a.data),
            comm=PETSc.COMM_SELF)
        objects.append(mat)
        mat.assemble()
        b = PETSc.Vec().createWithArray(np.asarray(rhs, dtype=np.float64).copy(), comm=PETSc.COMM_SELF)
        objects.append(b)
        x = b.duplicate()
        objects.append(x)
        solver = PETSc.KSP().create(PETSc.COMM_SELF)
        objects.append(solver)
        solver.setType('preonly')
        solver.getPC().setType('lu')
        solver.getPC().setFactorSolverType('mumps')
        solver.setOperators(mat)
        solver.setErrorIfNotConverged(True)
        solver.solve(b, x)
        if solver.getConvergedReason() <= 0:
            raise RuntimeError('MUMPS flow linear solve did not converge')
        value = x.getArray(readonly=True).copy()
        residual = np.linalg.norm(a @ value-rhs)/max(np.linalg.norm(rhs), 1.)
        if not np.isfinite(value).all() or not np.isfinite(residual) or residual > 1e-10:
            raise RuntimeError('MUMPS flow linear residual exceeds 1e-10')
        return value
    finally:
        for obj in reversed(objects):
            obj.destroy()
