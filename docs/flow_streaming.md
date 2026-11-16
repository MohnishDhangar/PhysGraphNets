# Streamed CPU flow trajectories

The fixed-step solver can now run in bounded-size segments, with recoverable
checkpoints and on-disk fields. Numerical integration stays in the existing FEM
solver. This implementation is for POSIX CPU hosts (including the local Linux and
Linux cloud environments); it adds no dependencies.

Run a manufactured patch experiment:

```bash
python scripts/run_flow_stream.py --output outputs/stream_patch \
  --n 4 --dt 0.025 --total-steps 12 --chunk-steps 3 --traction-tag right
python scripts/run_flow_stream.py --output outputs/stream_patch \
  --n 4 --dt 0.025 --total-steps 20 --chunk-steps 3 --traction-tag right --resume
```

`--total-steps` is the absolute target from the original initial time. The second
command extends the existing solution by eight steps. The chunk size may change
on resume; the physical problem, mesh, timestep, scheme and other numerical
settings must match. A completed resume is a validated no-op. Resume requires the
same numerical source and runtime versions. The original experiment manifest is
retained; the latest manifest records the current attempt, progress, measured
errors, source identity and discrete checks.

The default accelerating-shear case with linear amplitude is an exact patch test.
The command also supports vortex and translated-vortex manufactured cases and
exponential amplitude. Errors are reported against analytical manufactured values.
Passing discrete checks is not a mesh/time refinement qualification. Cylinder
benchmark qualification remains a separate task.

## Python interface

```python
from physgraph.experiments.flow_stream import stream_navier_stokes

archive = stream_navier_stokes(
    mesh, forcing, boundary_velocity, initial_velocity,
    output='outputs/trajectory', total_steps=1600, chunk_steps=50,
    dt=0.005, problem_id='immutable-case-and-parameters:v1',
)
for segment in archive.iter_segments():
    process(segment)  # consume and release; do not accumulate large segments
```

`solver_options` accept the original numerical controls, boundary partition,
viscous form and optional step observer. Streaming owns checkpoint publication
and the restart state. A step observer sees converged steps; a `chunk_callback`
sees committed segments. Initial data and arbitrary physical callbacks are not
serialized: the caller must give them a new `problem_id` when their meaning
changes. See [the restart contract](flow_restart.md).

Each segment includes its first velocity, newly solved physical pressure and
reactions, global diagnostic step/time values, and the velocity before the segment
when needed for BDF2 energy. Shared endpoint velocities are duplicated on disk;
pressure entries are never duplicated or fabricated at a segment start.

## Commit and recovery

The archive has one static `mesh.npz`, `segment_XXXXXXXX.npz` files, matching
`state_XXXXXXXX.npz` checkpoints, and a small `index.json`. Numeric arrays and JSON
are stored without pickle. Each archive file has an integrity checksum. The index
identifies a committed contiguous prefix; readers keep a snapshot of that prefix.

A writer takes an OS process lock, writes and syncs the segment and checkpoint,
then atomically publishes and syncs the new index. Concurrent writers are rejected.
An abrupt process exit releases the lock automatically. Uncommitted trailing files
are ignored and may be overwritten during resume. A failure during a segment can
require recomputing at most `chunk_steps` steps. Corrupted committed data is an
error; it is never silently skipped. Resume audits the committed prefix before
solving further. This scan is bounded in memory but takes time proportional to
archive size. Storage must have normal POSIX locking, rename and fsync semantics.

Peak trajectory-field memory scales with chunk size and mesh DOFs, independently
of the total requested time steps. Reading/verification may hold a small constant
number of segments, and compression uses temporary buffers. Mesh and sparse
solver storage remain in memory. Assembly repeats at chunk boundaries; this
tradeoff preserves the proven integrator and can be profiled separately. This is
not a claim that the complete solver has constant memory as the mesh is refined.

Read-only segment iteration can inspect older numerical sources. Resuming or
`validate()` also checks the final checkpoint's source identity. Independent
FEniCSx comparison still requires a complete trajectory starting from its own
projection; streaming does not replace independent histories with local FEM data.
A streaming independent-verification adapter and cylinder study CLI integration
remain to be implemented before long benchmark studies use this archive format.
