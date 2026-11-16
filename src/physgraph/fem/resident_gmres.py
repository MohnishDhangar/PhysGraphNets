"""Restarted right-preconditioned GMRES with explicit true-residual checks.

Vector storage/operations belong to the supplied algebra. Only the small
Hessenberg least-squares problem and scalar convergence decisions run here.
"""
from time import perf_counter
import numpy as np


def resident_gmres(algebra, rhs, *, rtol=1e-12, restart=30, max_iterations=120,
                   timeout_seconds=60., workspace=None):
    if not 0 < rtol <= 1e-10 or restart < 1 or max_iterations < 1 or timeout_seconds <= 0:
        raise ValueError('invalid GMRES limits')
    start = perf_counter()
    if workspace is None:
        workspace={}
    if not workspace:
        workspace.update(x=algebra.zeros(),residual=algebra.zeros(),
            basis=[algebra.zeros() for _ in range(restart+1)],
            preconditioned=[algebra.zeros() for _ in range(restart)],
            work=algebra.zeros(),correction=algebra.zeros(),restart=restart)
    if workspace['restart']!=restart:
        raise ValueError('workspace restart size mismatch')
    x,residual=workspace['x'],workspace['residual']
    algebra.zero(x);algebra.assign(residual,rhs)
    norm_b = algebra.norm(rhs)
    if norm_b == 0:
        return x, dict(iterations=0, relative_residual=0., residual_history=[0.])
    target = rtol*norm_b
    history, iterations = [], 0
    # Allocate once; restarted cycles reuse these full-size workspaces.
    basis,preconditioned=workspace['basis'],workspace['preconditioned']
    work,correction=workspace['work'],workspace['correction']
    while iterations < max_iterations:
        beta = algebra.norm(residual)
        history.append(beta/norm_b)
        if not np.isfinite(beta):
            raise RuntimeError('nonfinite GMRES true residual')
        if beta <= target:
            return x, dict(iterations=iterations,relative_residual=beta/norm_b,residual_history=history)
        algebra.assign(basis[0],residual)
        algebra.scale(basis[0],1./beta)
        h = np.zeros((restart+1,restart))
        e = np.zeros(restart+1); e[0] = beta
        for j in range(min(restart,max_iterations-iterations)):
            if perf_counter()-start > timeout_seconds:
                raise RuntimeError('GMRES wall-time limit exceeded')
            algebra.precondition(basis[j],preconditioned[j])
            algebra.matvec(preconditioned[j],work)
            # Twice-modified Gram-Schmidt protects orthogonality in float64.
            for _ in range(2):
                for i in range(j+1):
                    coefficient = algebra.dot(basis[i],work)
                    h[i,j] += coefficient
                    algebra.axpy(work,basis[i],-coefficient)
            h[j+1,j] = algebra.norm(work)
            if not np.isfinite(h[:j+2,j]).all():
                raise RuntimeError('nonfinite GMRES Arnoldi basis')
            if h[j+1,j] > 0:
                algebra.assign(basis[j+1],work)
                algebra.scale(basis[j+1],1./h[j+1,j])
            y = np.linalg.lstsq(h[:j+2,:j+1],e[:j+2],rcond=None)[0]
            estimate = np.linalg.norm(e[:j+2]-h[:j+2,:j+1] @ y)
            iterations += 1
            if estimate <= target or h[j+1,j] == 0 or j == restart-1 or iterations == max_iterations:
                algebra.zero(correction)
                for k in range(j+1):
                    algebra.axpy(correction,preconditioned[k],float(y[k]))
                algebra.axpy(x,correction,1.)
                algebra.matvec(x,work)
                algebra.assign(residual,rhs)
                algebra.axpy(residual,work,-1.)
                break
    relative = algebra.norm(residual)/norm_b
    history.append(relative)
    if not np.isfinite(relative) or relative > rtol:
        raise RuntimeError(f'GMRES failed after {iterations} iterations: true relative residual {relative:.3e}')
    return x, dict(iterations=iterations,relative_residual=relative,residual_history=history)
