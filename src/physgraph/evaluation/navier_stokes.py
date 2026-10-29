"""Trajectory accuracy, momentum balance and closed-wall energy diagnostics."""
import numpy as np
from physgraph.evaluation.stokes import stokes_errors, stokes_boundary_integrals
from physgraph.fem.flow_operators import FlowOperators


def trajectory_metrics(trajectory, case):
    frames = []
    for step, time in enumerate(trajectory.times[1:], 1):
        frame = trajectory.frame(step)
        reference = case.at(time)
        frames.append({**frame.diagnostics, **stokes_errors(frame, reference),
                       'boundary_integrals': stokes_boundary_integrals(frame, reference)})
    maxima = {key: max(abs(row[key]) for row in frames) for key in
        ('relative_residual', 'max_boundary_error', 'weak_divergence_residual',
         'boundary_flux', 'algebraic_force_balance')}
    maxima['pressure_mean'] = max(abs(row['pressure_mean']-case.pressure_offset) for row in frames)
    maxima['net_integrated_flux'] = max(abs(row['boundary_integrals']['net_flux']) for row in frames)
    return {'initial': trajectory.initial_diagnostics, 'maxima': maxima, 'frames': frames,
            'final_errors': {key: frames[-1][key] for key in
                ('velocity_l2', 'velocity_h1_seminorm', 'pressure_l2', 'divergence_l2')}}


def closed_wall_energy(trajectory):
    """BE physical-energy / BDF2 G-energy identities for zero forcing and walls.

    Includes advective power: weakly divergence-free P2 velocity does not make
    this term identically zero. This diagnostic asserts no unconditional stability.
    """
    ops = FlowOperators(trajectory.space, trajectory.viscosity)
    vectors = [u.T.ravel() for u in trajectory.velocity]
    energy = [float(.5*u @ ops.mass @ u) for u in vectors]
    rows = []
    for step, state in enumerate(trajectory.diagnostics, 1):
        if (np.max(np.abs(trajectory.velocity[step][trajectory.space.boundary])) > 1e-12
                or state['body_load_norm'] > 1e-12 or state['traction_load_norm'] > 1e-12):
            raise ValueError('closed-wall energy diagnostic requires zero walls and zero applied loads')
        u, previous = vectors[step], vectors[step-1]
        dt = trajectory.times[step]-trajectory.times[step-1]
        if state['method'] == 'be':
            rate = (energy[step]-energy[step-1])/dt
            diff = u-previous
            dissipation = float(.5*diff @ ops.mass @ diff/dt)
            modified_energy = energy[step]
        else:
            older = vectors[step-2]
            a, b = 2*u-previous, 2*previous-older
            modified_energy = float(.25*(u @ ops.mass @ u+a @ ops.mass @ a))
            previous_modified = float(.25*(previous @ ops.mass @ previous+b @ ops.mass @ b))
            rate = (modified_energy-previous_modified)/dt
            diff = u-2*previous+older
            dissipation = float(.25*diff @ ops.mass @ diff/dt)
        balance = rate+dissipation+state['viscous_dissipation']+state['convective_power']
        rows.append(dict(time=state['time'], kinetic_energy=energy[step],
            modified_energy=modified_energy, numerical_dissipation=dissipation,
            viscous_dissipation=state['viscous_dissipation'], convective_power=state['convective_power'],
            energy_balance=float(balance)))
    return {'initial_energy': energy[0], 'rows': rows,
            'max_energy_increase': float(np.max(np.diff(energy))),
            'max_absolute_balance': max(abs(row['energy_balance']) for row in rows)}
