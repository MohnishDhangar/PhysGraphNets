# Fixed-step flow checkpoint and restart

`solve_navier_stokes` can expose its complete continuation state when given a
nonempty `problem_id`. This identifier belongs to the caller: change it whenever
forcing, initial data, boundary callbacks, or their parameters change. Python
callbacks are not serialized or automatically certified equivalent.

```python
from physgraph.io.flow_checkpoint import save_flow_checkpoint, load_flow_checkpoint

# Supply the same mesh, callbacks and solver options on both calls.
first = solve_navier_stokes(
    mesh, forcing, boundary_velocity, initial_velocity,
    dt=0.005, steps=100, problem_id='case-and-parameters:v1',
    checkpoint_callback=lambda state: save_flow_checkpoint('latest.npz', state),
)
continued = solve_navier_stokes(
    mesh, forcing, boundary_velocity, initial_velocity,
    dt=0.005, steps=100, problem_id='case-and-parameters:v1',
    restart_state=load_flow_checkpoint('latest.npz'),
)
```

`steps` is the number of additional steps. Keep `initial_time` at the original
simulation origin. The solver computes times from the global step index and
retains both BDF2 velocity histories, the physical pressure/Newton guess, and any
pressure-gauge multiplier. A resumed BDF2 solve does not repeat BE startup.
Callbacks and diagnostic step numbers use global indices.

The returned trajectory is a segment: its first velocity is the saved restart
velocity. Only newly solved steps have physical pressure and reaction entries.
Thus `segment.pressure` corresponds to `segment.times[1:]`, and `frame(0)` is
unavailable. To join segments, drop the duplicated first velocity/time of every
segment after the first; concatenate their pressure and reaction arrays directly.
Initial diagnostics describe the original constrained projection.

Checkpoint writes use a temporary file, file fsync, and atomic replacement. The
NPZ contains numeric arrays and JSON, never pickled Python objects. Loading checks
its checksum and source identity. Continuation additionally rejects incompatible
mesh/DOF ordering, numerical runtime versions, fixed timestep, viscosity, time
origin, scheme, quadrature, nonlinear controls, viscous form, pressure constraint,
boundary partition, or caller-owned problem identity. It verifies state shapes,
finite float64 values, Newton/velocity consistency and the boundary values of
both history states. A matching problem identifier cannot detect an incorrectly
reused identifier with changed interior forcing.

A callback receives detached copies after a converged step. If it raises, the
solve stops and the last successfully written checkpoint can be resumed. Checkpoint
files carry state, not the complete trajectory: retain prior trajectory segments
separately when their physical pressure, reactions or histories are needed.
The current in-memory trajectory allocation is unchanged; bounded-memory output
is a separate next deliverable.

Exact continuation is tested against uninterrupted runs in the same CPU
numerical environment. Cross-hardware or changed-library bitwise reproducibility
is not claimed. A source mismatch requires a new run, not bypassing the check.

Resumed segments also retain `velocity_before_segment`, so closed-wall BDF2
G-energy diagnostics use the correct pre-segment history. Trajectory exports
include this array (empty for an original segment) and `segment_start_step` in
metadata. Independent FEniCSx comparison rejects a resumed segment alone: join
the complete original trajectory first, preserving its original initial velocity
and all diagnostic rows. This keeps the independent solver's own projection and
time history as the reference.
