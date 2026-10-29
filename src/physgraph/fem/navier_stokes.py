"""Fully implicit CPU Taylor-Hood Navier-Stokes on fixed triangular meshes."""
from collections.abc import Mapping
from dataclasses import dataclass
import warnings
import numpy as np
from scipy.sparse.linalg import MatrixRankWarning, spsolve
from physgraph.fem.stokes import StokesResult, TaylorHoodSpace, taylor_hood_space
from physgraph.fem.stokes_boundary import stokes_partition, stokes_velocity_values
from physgraph.fem.flow_operators import FlowOperators


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

    def frame(self, step):
        """Physical solution at step >=1; no fabricated physical pressure at t0."""
        if type(step) is not int or not 1 <= step < len(self.times):
            raise ValueError('physical pressure is available only at solved steps >= 1')
        return StokesResult(self.space, self.velocity[step], self.pressure[step-1],
            self.diagnostics[step-1], self.viscosity, 'symmetric_gradient',
            self.dirichlet_tags, self.reactions[step-1])


def _direct(matrix, rhs):
    with warnings.catch_warnings():
        warnings.simplefilter('error', MatrixRankWarning)
        value = spsolve(matrix, rhs)
    if not np.isfinite(value).all():
        raise RuntimeError('nonfinite flow linear solution')
    return value


def solve_navier_stokes(mesh, forcing, boundary_velocity, initial_velocity, *, dt, steps,
                        viscosity=.1, scheme='bdf2', dirichlet_tags=None, tractions=None,
                        initial_time=0., quadrature_order=9, newton_tolerance=1e-12,
                        max_iterations=20):
    """Integrate with BE or fixed-step BDF2 (one BE startup step).

    forcing/boundary_velocity accept (points,time); initial_velocity accepts points.
    Boundary velocity may instead map tags to time-dependent callbacks. Tractions
    map tags to callbacks (points,outward_normals,time). Failed nonlinear solves
    raise; no unconverged state is returned. No adaptive timestep or stabilization.
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
    space = taylor_hood_space(mesh)
    prescribed, facets, traction_data, tags = stokes_partition(space, dirichlet_tags, tractions)
    ops = FlowOperators(space, viscosity, quadrature_order)
    nv, npres = ops.nv, ops.np
    mean_gauge = not traction_data
    fixed = np.r_[prescribed, prescribed+nv]
    total = 2*nv+npres+int(mean_gauge)
    free = np.setdiff1d(np.arange(total), fixed)
    times = initial_time+dt*np.arange(steps+1)
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
    projected[free] = _direct(projection[free][:, free], lifted)
    projection_residual = np.linalg.norm((projection @ projected-projection_rhs)[free])/max(np.linalg.norm(lifted), 1.)
    if projection_residual > 1e-10:
        raise RuntimeError('initial divergence-free projection failed')
    initial_diagnostics = dict(projection_residual=float(projection_residual),
        weak_divergence_residual=float(np.linalg.norm(ops.divergence @ projected[:2*nv])),
        max_boundary_error=float(np.max(np.abs(projected[fixed]-g0))),
        pressure_at_initial_time='not provided; projection multiplier is not physical pressure')
    velocities = [projected[:2*nv].reshape(2, nv).T.copy()]
    pressures, reactions, diagnostics = [], [], []
    previous = projected[:2*nv].copy()
    older = None
    guess = projected.copy()
    guess[2*nv:] = 0.  # discard projection multipliers
    for step, time in enumerate(times[1:], 1):
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
            delta = _direct(jacobian[free][:, free], -r[free])
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
    return FlowTrajectory(space, times, np.array(velocities), np.array(pressures),
        np.array(reactions), diagnostics, initial_diagnostics, viscosity, tags, scheme)
