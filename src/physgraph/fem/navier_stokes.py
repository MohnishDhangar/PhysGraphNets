"""Fully implicit CPU Taylor-Hood Navier-Stokes on fixed triangular meshes."""
from collections.abc import Mapping
from dataclasses import dataclass
from copy import deepcopy
from physgraph.core.flow_state import FlowRestartState
from physgraph.io.flow_checkpoint import flow_source_fingerprint, mesh_fingerprint, flow_runtime_identity
import warnings
import numpy as np
from scipy.sparse.linalg import MatrixRankWarning, spsolve
from physgraph.fem.stokes import StokesResult, TaylorHoodSpace, taylor_hood_space
from physgraph.fem.stokes_boundary import stokes_partition, stokes_velocity_values
from physgraph.fem.flow_operators import FlowOperators
from physgraph.fem.flow_linear_solver import backend_identity, mumps_solve


@dataclass
class FlowTrajectory:
    space: TaylorHoodSpace
    times: np.ndarray
    velocity: np.ndarray
    pressure: np.ndarray
    reactions: np.ndarray
    diagnostics: list
    initial_diagnostics: dict
    viscosity: float
    dirichlet_tags: tuple | None
    scheme: str
    viscous_form: str = 'symmetric_gradient'
    restart_state: FlowRestartState | None = None
    velocity_before_segment: np.ndarray | None = None
    linear_solver_records: list | None = None

    def frame(self, step):
        """Physical solution at step >=1; no fabricated physical pressure at t0."""
        if type(step) is not int or not 1 <= step < len(self.times):
            raise ValueError('physical pressure is available only at solved steps >= 1')
        return StokesResult(self.space, self.velocity[step], self.pressure[step-1],
            self.diagnostics[step-1], self.viscosity, self.viscous_form,
            self.dirichlet_tags, self.reactions[step-1])


def _direct(matrix, rhs):
    with warnings.catch_warnings():
        warnings.simplefilter('error', MatrixRankWarning)
        value = spsolve(matrix, rhs)
    if not np.isfinite(value).all():
        raise RuntimeError('nonfinite flow linear solution')
    return value


def _restart_contract(space, tags, traction_data, options):
    """One contract builder shared by integration and no-op archive resume checks."""
    return dict(
        source=flow_source_fingerprint(), mesh=mesh_fingerprint(space), problem_id=options['problem_id'],
        runtime=flow_runtime_identity(),
        dt=float(options['dt']), initial_time=float(options['initial_time']),
        viscosity=float(options['viscosity']), scheme=options['scheme'],
        quadrature_order=options['quadrature_order'], newton_tolerance=float(options['newton_tolerance']),
        max_iterations=options['max_iterations'], viscous_form=options['viscous_form'],
        dirichlet_tags=None if tags is None else list(tags), traction_tags=sorted(traction_data),
        mean_gauge=not traction_data, precision='float64',
        linear_backend=backend_identity(options.get('linear_backend', 'scipy')))


def solve_navier_stokes(mesh, forcing, boundary_velocity, initial_velocity, *, dt, steps,
                        viscosity=.1, scheme='bdf2', dirichlet_tags=None, tractions=None,
                        initial_time=0., quadrature_order=9, newton_tolerance=1e-12,
                        max_iterations=20, viscous_form='symmetric_gradient', step_callback=None,
                        problem_id=None, restart_state=None, checkpoint_callback=None,
                        linear_backend='scipy'):
    """Integrate BE/BDF2 flow with an explicit CPU or experimental GPU backend.

    CPU assembly, physical definitions, Newton residuals and tolerances are shared.
    rocm_gmres performs CPU ILUT setup and GPU-resident Krylov/triangular operations;
    setup and iteration diagnostics are returned in linear_solver_records.
    Restart retains the original time origin and both BDF2 histories. A GPU
    preconditioner is rebuilt at timestep boundaries, including after restart.
    """
    arguments = locals().copy()
    if linear_backend not in ('rocm_gmres','rocm_cached','cpu_cached'):
        return _integrate_navier_stokes(**arguments)
    backend_identity(linear_backend)
    from physgraph.fem.rocm_flow_solver import RocmFlowSolver
    if linear_backend=='rocm_gmres':
        solver=RocmFlowSolver()
    else:
        from physgraph.fem.cached_flow_solver import CachedFlowSolver
        solver=CachedFlowSolver('gpu' if linear_backend=='rocm_cached' else 'cpu')
    try:
        result = _integrate_navier_stokes(**arguments, _linear_solve=solver)
        result.linear_solver_records = solver.records
        return result
    finally:
        solver.close()


def _integrate_navier_stokes(mesh, forcing, boundary_velocity, initial_velocity, *, dt, steps,
                        viscosity=.1, scheme='bdf2', dirichlet_tags=None, tractions=None,
                        initial_time=0., quadrature_order=9, newton_tolerance=1e-12,
                        max_iterations=20, viscous_form='symmetric_gradient', step_callback=None,
                        problem_id=None, restart_state=None, checkpoint_callback=None,
                        linear_backend='scipy', _linear_solve=None):
    """Integrate with BE or fixed-step BDF2 (one BE startup step).

    forcing/boundary_velocity accept (points,time); initial_velocity accepts points.
    Boundary velocity may instead map tags to time-dependent callbacks. Tractions
    map tags to callbacks (points,outward_normals,time). Failed nonlinear solves
    raise; no unconverged state is returned. No adaptive timestep or stabilization.

    Set problem_id to an immutable, caller-owned identity of all physical callbacks
    and parameters to enable restart states. Callables are not serialized: the
    caller must change this identity when forcing, boundary, or initial data change.
    restart_state continues for `steps` additional steps; initial_time remains the
    original time origin. Both BDF2 histories and the Newton guess are preserved.
    checkpoint_callback receives a detached state after each converged step.
    Returned segment index zero has velocity only, even when resuming.
    linear_backend='mumps' opts into optional real-float64 PETSc/MUMPS on CPU.
    Set BLAS/OpenMP thread limits before starting Python. SciPy is the default.
    """
    if not np.isfinite(dt) or dt <= 0 or not np.isfinite(initial_time) or initial_time < 0:
        raise ValueError('dt must be positive and initial time nonnegative, both finite')
    if type(steps) is not int or steps < 1 or type(max_iterations) is not int or max_iterations < 1:
        raise ValueError('steps and max_iterations must be positive integers')
    if scheme not in ('be', 'bdf2'):
        raise ValueError('scheme must be be or bdf2')
    if not np.isfinite(newton_tolerance) or not 0 < newton_tolerance <= 1e-8:
        raise ValueError('Newton tolerance must be positive and <= 1e-8')
    if not callable(forcing) or not callable(initial_velocity):
        raise ValueError('forcing and initial velocity must be callable')
    if step_callback is not None and not callable(step_callback):
        raise ValueError('step_callback must be callable')
    if checkpoint_callback is not None and not callable(checkpoint_callback):
        raise ValueError('checkpoint_callback must be callable')
    if problem_id is not None and (not isinstance(problem_id, str) or not problem_id.strip()):
        raise ValueError('problem_id must be a nonempty string')
    if (restart_state is not None or checkpoint_callback is not None) and problem_id is None:
        raise ValueError('restart and checkpointing require explicit problem_id')
    if restart_state is not None and not isinstance(restart_state, FlowRestartState):
        raise ValueError('restart_state must be a FlowRestartState')
    backend_identity(linear_backend)
    linear_solve = _linear_solve if _linear_solve is not None else (_direct if linear_backend == 'scipy' else mumps_solve)
    space = taylor_hood_space(mesh)
    prescribed, facets, traction_data, tags = stokes_partition(space, dirichlet_tags, tractions)
    ops = FlowOperators(space, viscosity, quadrature_order, viscous_form)
    nv, npres = ops.nv, ops.np
    mean_gauge = not traction_data
    fixed = np.r_[prescribed, prescribed+nv]
    total = 2*nv+npres+int(mean_gauge)
    free = np.setdiff1d(np.arange(total), fixed)
    contract = None if problem_id is None else _restart_contract(space, tags, traction_data, locals())
    start_step = 0
    if restart_state is not None:
        if restart_state.contract != contract:
            raise ValueError('incompatible restart contract (source, mesh, problem or solver settings)')
        if type(restart_state.step) is not int or restart_state.step < 1:
            raise ValueError('restart requires a completed positive step')
        start_step = restart_state.step
        for name, size in (('previous', 2*nv), ('older', 2*nv), ('guess', total)):
            value = getattr(restart_state, name)
            if (not isinstance(value, np.ndarray) or value.dtype != np.float64
                    or value.shape != (size,) or not np.isfinite(value).all()):
                raise ValueError('invalid restart state vector: '+name)
        if not np.array_equal(restart_state.previous, restart_state.guess[:2*nv]):
            raise ValueError('restart velocity and Newton guess disagree')
    times = initial_time+dt*np.arange(start_step, start_step+steps+1)
    if not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
        raise ValueError('time grid must be finite and strictly increasing')

    def boundary(time):
        if callable(boundary_velocity):
            data = lambda x: boundary_velocity(x, time)
        elif isinstance(boundary_velocity, Mapping) and all(callable(f) for f in boundary_velocity.values()):
            data = {tag: (lambda x, f=f: f(x, time)) for tag, f in boundary_velocity.items()}
        else:
            raise ValueError('boundary velocity must be callable or a mapping of callbacks')
        values = stokes_velocity_values(space, prescribed, facets, tags, data)
        lift = np.zeros(2*nv)
        lift[fixed] = values.T.ravel()
        if mean_gauge and abs(float(np.sum(ops.divergence @ lift))) > 1e-11*max(1., np.linalg.norm(values)):
            raise ValueError(f'incompatible prescribed boundary flux at time {time}')
        return values.T.ravel()

    if restart_state is None:
        # Constrained L2 projection avoids injecting interpolation divergence at t0.
        initial_values = np.asarray(initial_velocity(space.points[prescribed]), dtype=float)
        g0 = boundary(initial_time)
        if (initial_values.shape != (len(prescribed), 2) or not np.isfinite(initial_values).all()
                or not np.allclose(initial_values.T.ravel(), g0, rtol=1e-11, atol=1e-12)):
            raise ValueError('initial velocity must match prescribed velocity at initial time')
        projection = ops.saddle(ops.mass, mean_gauge)
        projected = np.zeros(total)
        projected[fixed] = g0
        initial_load = ops.volume_load(initial_velocity)
        projection_rhs = np.r_[initial_load, np.zeros(npres+int(mean_gauge))]
        lifted = (projection_rhs-projection @ projected)[free]
        projected[free] = linear_solve(projection[free][:, free], lifted)
        projection_residual = np.linalg.norm((projection @ projected-projection_rhs)[free])/max(np.linalg.norm(lifted), 1.)
        if projection_residual > 1e-10:
            raise RuntimeError('initial divergence-free projection failed')
        initial_diagnostics = dict(projection_residual=float(projection_residual),
            weak_divergence_residual=float(np.linalg.norm(ops.divergence @ projected[:2*nv])),
            max_boundary_error=float(np.max(np.abs(projected[fixed]-g0))),
            pressure_at_initial_time='not provided; projection multiplier is not physical pressure')
        previous = projected[:2*nv].copy()
        older = None
        guess = projected.copy()
        guess[2*nv:] = 0.  # discard projection multipliers
    else:
        previous, older, guess = (restart_state.previous.copy(), restart_state.older.copy(),
                                  restart_state.guess.copy())
        initial_diagnostics = deepcopy(restart_state.initial_diagnostics)
        for state_velocity, at_step in ((previous, start_step), (older, start_step-1)):
            if not np.array_equal(state_velocity[fixed], boundary(initial_time+dt*at_step)):
                raise ValueError('restart state does not match prescribed boundary velocity')
    velocities = [previous.reshape(2, nv).T.copy()]
    pressures, reactions, diagnostics = [], [], []
    final_state = None
    velocity_before_segment = None if older is None else older.reshape(2, nv).T.copy()
    for step, time in enumerate(times[1:], start_step+1):
        if hasattr(linear_solve, 'begin_step'):
            linear_solve.begin_step(step)
        alpha = 1. if scheme == 'be' or step == 1 else 1.5
        history = previous/dt if alpha == 1. else (2*previous-.5*older)/dt
        linear = ops.saddle(alpha/dt*ops.mass+ops.viscous, mean_gauge)
        body = ops.volume_load(lambda x: forcing(x, time))
        applied = ops.traction_load(facets, {tag: (lambda x, n, f=f: f(x, n, time))
                                           for tag, f in traction_data.items()})
        rhs = np.r_[body+applied+ops.mass @ history, np.zeros(npres+int(mean_gauge))]
        prescribed_values = boundary(time)
        guess[fixed] = prescribed_values
        lift = np.zeros(total)
        lift[fixed] = prescribed_values
        scale = max(np.linalg.norm((rhs-linear @ lift)[free]), 1.)

        def residual(value):
            answer = linear @ value-rhs
            answer[:2*nv] += ops.convection(value[:2*nv])
            return answer

        damping = []
        for iteration in range(max_iterations+1):
            r = residual(guess)
            norm = np.linalg.norm(r[free])
            if norm <= newton_tolerance*scale:
                break
            if iteration == max_iterations:
                raise RuntimeError(f'Navier-Stokes Newton failed at step {step}, t={time}: residual={norm/scale:.3e}')
            _, convection_jacobian = ops.convection(guess[:2*nv], jacobian=True)
            jacobian = ops.saddle(alpha/dt*ops.mass+ops.viscous+convection_jacobian, mean_gauge)
            delta = linear_solve(jacobian[free][:, free], -r[free])
            accepted = False
            for backtrack in range(13):
                fraction = .5**backtrack
                candidate = guess.copy()
                candidate[free] += fraction*delta
                candidate_norm = np.linalg.norm(residual(candidate)[free])
                if np.isfinite(candidate_norm) and (candidate_norm <= (1-1e-4*fraction)*norm
                                                    or candidate_norm <= newton_tolerance*scale):
                    guess = candidate
                    damping.append(fraction)
                    accepted = True
                    break
            if not accepted:
                raise RuntimeError(f'Navier-Stokes Newton line search failed at step {step}, t={time}')
        u, p = guess[:2*nv], guess[2*nv:2*nv+npres]
        momentum = ops.mass @ (alpha*u/dt-history)+ops.convection(u)
        inertial_force = momentum.reshape(2, nv).sum(axis=1)
        body_force = body.reshape(2, nv).sum(axis=1)
        applied_force = applied.reshape(2, nv).sum(axis=1)
        nodal_reactions = r[:2*nv].reshape(2, nv).T.copy()
        reaction = nodal_reactions[prescribed].sum(axis=0)
        diagnostics.append(dict(time=float(time), step=step, method='be' if alpha == 1. else 'bdf2',
            relative_residual=float(norm/scale), newton_iterations=iteration, newton_damping=damping,
            max_boundary_error=float(np.max(np.abs(guess[fixed]-prescribed_values))),
            weak_divergence_residual=float(np.linalg.norm(ops.divergence @ u)),
            boundary_flux=-float(np.sum(ops.divergence @ u)),
            pressure_mean=float(ops.pressure_integrals @ p/ops.pressure_integrals.sum()),
            pressure_constraint='mean_zero' if mean_gauge else 'traction',
            gauge_multiplier=float(guess[-1]) if mean_gauge else None,
            body_force=body_force.tolist(), prescribed_traction_force=applied_force.tolist(),
            body_load_norm=float(np.linalg.norm(body)), traction_load_norm=float(np.linalg.norm(applied)),
            inertial_force=inertial_force.tolist(), dirichlet_reaction_on_fluid=reaction.tolist(),
            algebraic_force_balance=float(np.linalg.norm(body_force+applied_force+reaction-inertial_force)),
            kinetic_energy=float(.5*u @ (ops.mass @ u)), viscous_dissipation=float(u @ (ops.viscous @ u)),
            convective_power=float(u @ ops.convection(u)), velocity_nodes=nv, pressure_nodes=npres,
            triangles=len(mesh.triangles)))
        velocities.append(u.reshape(2, nv).T.copy())
        pressures.append(p.copy())
        reactions.append(nodal_reactions)
        older, previous = previous, u.copy()
        if contract is not None:
            final_state = FlowRestartState(deepcopy(contract), step, previous.copy(), older.copy(),
                                           guess.copy(), deepcopy(initial_diagnostics))
            if checkpoint_callback is not None:
                checkpoint_callback(deepcopy(final_state))
        if step_callback is not None:
            step_callback(step, float(time))
    return FlowTrajectory(space, times, np.array(velocities), np.array(pressures),
        np.array(reactions), diagnostics, initial_diagnostics, viscosity, tags, scheme, viscous_form,
        final_state, velocity_before_segment)
