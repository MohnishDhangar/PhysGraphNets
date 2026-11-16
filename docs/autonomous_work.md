# Autonomous development handoff

User authorization: verify the running full qualification study, then continue
the prioritized objectives and required tests without routine confirmation while
the user's current five-hour usage allowance permits. The remaining account
allowance is not visible to the agent; platform limits govern availability.
An ongoing goal is active in this thread. Do not interpret this as a request to
spend five wall-clock hours or invent a token budget.

## Completed qualification evidence

- Study: `outputs/cylinder_full_qualification_20261112/manifest.json`.
- Log: `outputs/cylinder_full_qualification_20261112.log`.
- Live progress: child `progress.json` files under the study directory.
- Session 37801 is terminal (exit 1): all five trajectories finished; scientific
  acceptance failed spatial and temporal refinement. Independent comparison passed.
  Post-run artifact audit passed all five archives; see `artifact_audit.json` and
  `assessment.md` under the study directory. Elapsed study time: 15145.26 seconds.
- Declared meshes: 4/8/16 at dt=0.005; temporal steps 0.02/0.01/0.005 at level 16.
- Independently compare every finest-trajectory step with FEniCSx.
- Prior two-mesh evidence already failed spatial stability; preserve failures and
  do not relax acceptance or describe a passed benchmark candidate as qualification.
- The study finished with its source fingerprint preserved. Root source edits
  are now possible after reviewing staging differences. Preserve the completed
  artifacts and their original frozen source identity; never relabel them.

## Isolated preparation

`outputs/next_objectives_workspace/` contains a copy of the current package,
scripts, tests, configuration and docs. `staging_origin.json` records the origin.
It is ignored by Git and excluded from the running study's source fingerprint.
Use the existing root virtual-environment interpreter with PYTHONPATH pointing
to this copy's `src` and the staging directory as cwd. Confirm imported module
paths before testing. No environment installation or dependency change is needed.
Do not copy generated results or environments back into the main source tree.

Next preparatory work: checkpoint/restart and bounded-memory trajectory output.
Design restart around both BDF2 velocity histories, pressure/Newton guess, fixed
timestep, global step/time, mesh/DOF order, boundary/physics identity and precision.
Require split/uninterrupted equivalence for BE and BDF2, nonzero boundaries,
invalid/incompatible checkpoint rejection and preservation of pressure-time semantics.
Checkpoint and streamed I/O belong in I/O modules; the numerical solver owns
time integration. CLI scripts only orchestrate. This work supports larger
qualification studies and does not bypass unresolved numerical gates.

## Completion checks and integration order

1. Inspect study status, every run's checks, external comparison, refinement
   differences and failures. Verify source identities and archive consistency.
2. Record a report with Python, case, mesh/time sizes, errors and revision/hash.
3. Resolve failed qualification through justified refinement/diagnosis. Preserve
   unsuccessful studies; never substitute neural values for references.
4. Integrate reviewed staging changes only after the original run has finished.
   Preserve any intervening user edits. Run full pytest and the mesh convergence
   demo for numerical changes; record executed checks accurately.
5. Continue to triangular particle sampling/boundary policies, transient datasets
   and model qualification, profiling, and final viewer/release work as prerequisites
   permit. GPU dependencies/custom kernels require their separate phase decision.

Update this log with completed changes, exact test commands/results, active jobs,
and remaining gates so work can resume after interruption or a usage limit.

## Checkpoint/restart preparation (12 November, isolated copy)

Implemented in staging only:
- `src/physgraph/core/flow_state.py`: complete fixed-step temporal state.
- `src/physgraph/io/flow_checkpoint.py`: atomic pickle-free checkpoint save/load,
  checksums, numerical source and runtime identities, mesh/DOF fingerprint.
- `src/physgraph/fem/navier_stokes.py`: optional problem identity, restart state
  and detached checkpoint callbacks; global indices preserve BDF2 startup/history.
- `src/physgraph/evaluation/navier_stokes.py`: resumed BDF2 energy uses retained
  pre-segment velocity, not a wrapped local array index.
- `src/physgraph/verification/navier_stokes.py`: independent comparison requires
  the complete original trajectory and its independent initial projection.
- `src/physgraph/experiments/navier_stokes.py`: archive segment origin/history.
- `tests/test_flow_checkpoint.py`: 43 focused tests passed (3.88 seconds), including
  split/uninterrupted bitwise equality, BE/BDF2, nonzero and mixed boundaries,
  interruption recovery, failed atomic writes, incompatibility/corruption, energy.
- `docs/flow_restart.md`: API, physical identity responsibility and limitations.

Default solver parity was executed against the frozen root package in four
BE/BDF2 and Dirichlet/traction cases (translated vortex, n=2, dt=.03, four steps).
All time, velocity, pressure and reaction arrays were bitwise identical.
Evidence lives in staging `outputs/checkpoint_verification/`, including imported
package paths, revision/hash, errors and mesh sizes. The checkpoint feature does
not yet bound full trajectory memory or integrate with cylinder CLI/study restart.

The first full staging suite passed 449 tests with one expected dependency-path
skip (154.91 seconds). That result preceded the energy-history fix. The second full
suite completed with the final changes: **452 passed, 1 expected skip**, 132.09
seconds (session 97244 exited 0). The durable log is
`outputs/next_objectives_workspace/outputs/checkpoint_verification/pytest.log`.
Final convergence demo completed successfully at 8/16/32. Final test manifest:
`outputs/next_objectives_workspace/outputs/checkpoint_verification/checks_manifest.json`. Staging source hash for this version:
`1032ded0a08d4285fec3c8a29833d88c4756d43703f907dc9d4cfe3c439515c1`.

Next preparation: bounded-memory trajectory segments and recoverable index,
then experiment/CLI integration. Keep per-segment pressure-time semantics,
checkpoint consistency, physical/source identity and failure atomicity. Test
chunked/resumed trajectories against uninterrupted ones; avoid injecting local
numerical histories into the independent FEniCSx reference. Do not add dependencies.

## Segmented streaming preparation (12 November, isolated copy)

Added in staging only:
- `src/physgraph/io/flow_archive.py`: checksum-checked, pickle-free mesh/segments,
  per-segment checkpoints, constant-size commit index, POSIX exclusive writer
  locks, atomic publication and file/directory fsync. Resume audits the committed
  prefix. Uncommitted tails are ignored; corrupt committed data is rejected.
- `src/physgraph/experiments/flow_stream.py`: bounded-size solve orchestration,
  absolute total-step resume, source drift checks, and a manufactured experiment
  with original/latest manifests and analytical-error/discrete diagnostics.
- `scripts/run_flow_stream.py`: thin manufactured experiment CLI.
- `tests/test_flow_archive.py`: bitwise comparisons, mixed/nonzero boundaries,
  tagged/untagged meshes, short DFG cylinder with Laplacian outlet, interruption,
  real abrupt child-process exit, corrupt files, exclusive locking, partial
  checkpoint/index failures, temporal/pressure mismatch, source drift, bounded
  segment allocation and release, CLI provenance and resume.
- `docs/flow_streaming.md`: format, usage, guarantees, limits and remaining work.
- `_restart_contract` extracted in FEM solver so completed/no-op resume validates
  the same scientific contract as a new numerical step.

Initial full suite: 492 passed, 1 expected skip in 130.92 seconds. Subsequent
review found publication uncertainty after rename but before directory fsync:
writer must be closed/reopened rather than accept more appends after that error.
A guard and dedicated recovery test were added. Final full suite completed: **493 passed, 1 expected skip**, 117.45 seconds;
session 31971 exited 0. Log `outputs/next_objectives_workspace/outputs/stream_verification/pytest.log`.
Earlier log retained as `pytest_before_commit_guard.log`.

Actual CLI experiment: accelerating shear, linear time profile, n=4, dt=.025,
right traction with pressure offset 2.5, BDF2; first 12 steps in chunks of 3,
then resumed to 20 in chunks of 2. Passed discrete/patch checks; final errors
velocity L2 8.99e-16, velocity H1 1.57e-14, pressure L2 8.31e-15,
divergence L2 5.22e-15. Original and latest manifests plus all committed chunks
are under staging `outputs/stream_verification/patch/`. This is storage/manufactured
verification, not cylinder qualification. Default solver fields/reactions/times
remain bitwise identical to frozen root for all four parity cases.

Final staging source hash (after publication guard):
`6ad80d202ce6b44c4e4b3dc736677f36450a3122d0c205a664c6886c873c90f3`.
Final verification manifest was generated successfully by the output helper:
`outputs/next_objectives_workspace/outputs/stream_verification/checks_manifest.json`.
Mesh convergence and the completed-run CLI resume check also passed.
Remaining next work: adapt independent FEniCSx verification to consume segments
while retaining its own projection and full history across chunk boundaries;
then integrate streaming into the cylinder experiment/study orchestration.
Do not inject local FEM states into the independent solver. The original full qualification is now terminal; it used frozen root source,
not the staging package.


## Latest status check and post-run audit

User asked to check completion. Confirmed all launched study/regression/demo jobs
terminal. Qualification exited 1 because its declared spatial and temporal gates
failed; no solver crash. All discrete checks, finest published benchmark,
surface/reaction agreement, and full independent trajectory passed.

Saved-artifact audit completed in 13.54 seconds: all five NPZ archives scanned
with bounded buffers, CRC checked, numeric fields finite, positive triangle
areas, exact time/pressure alignment, expected shapes, matching source identities,
SHA-256 recorded, all refinement comparisons recomputed identically.
`outputs/cylinder_full_qualification_20261112/artifact_audit.json` and
`assessment.md` preserve findings. An initial audit-tool NumPy private-API error
was fixed to use public header readers; its log is retained separately and did
not affect the numerical run.

Finest spatial pair (8 -> 16): lift summary difference 5.503%, final pressure
6.327% (limit 3%), lift-history RMS 18.973%, pressure-history RMS 14.635% (limit 5%).
Finest temporal pair (.01 -> .005): lift summary 4.875%, final pressure 3.471%,
lift-history RMS 14.917%. Peak-time differences passed both finest pairs.
The fine benchmark alone cannot establish convergence. Do not relax thresholds.

Next actions: integrate reviewed, fully tested staging changes while preserving
all preexisting root edits; verify root imports/tests. Complete independent
streaming comparison and cylinder runner integration. Profile finer-mesh costs
and declare a new refinement experiment, preserving this failed study. No new
long numerical run has been started after the completed study.

## Performance steering and root integration

User asked why runs take 4+ hours and requested a faster development workflow.
Investigated recorded per-case timings and prior cProfile evidence. Study performed
6,000 local steps + 1,600 independent steps, 14,035 local Newton iterations +
3,951 independent Newton iterations. Fine-mesh system: 67,539 unknowns before
Dirichlet elimination. Existing ten-step mesh-8 profile: sparse direct solves
54.4% and convection/Jacobian 26.4% of local solver time. No measured large
end-to-end solver speedup is claimed. Five case timings were 4.22, 19.08,
149.76 (including independent verification), 29.35 and 49.98 minutes.

Confirmed root source still matched frozen study, then integrated 13 reviewed
source/script/test files and the two restart/streaming docs from staging. Existing
root files were backed up under `outputs/stream_integration_20261112/original/`;
per-file hashes and full source identity are in `integration.json`. Preserved all
other root edits. Staging remains as a prior copy; ROOT IS NOW THE WORKING SOURCE.

Added `scripts/check_cylinder_flow.py`: explicit mesh-2, dt=.02, final-time .2
check with optional independent verification (10+10 steps). Added non-numerical
phase timers in `experiments/cylinder_flow.py`: meshing, local solve, archive
write, force/pressure evaluation, independent comparison. Future qualification
runs will provide an actual stage breakdown instead of an inferred split.

Executed new root smoke command with --fenicsx: passed all discrete/independent
checks in 2.893 seconds inside the experiment (interpreter startup excluded).
Local solve .526s, independent comparison 2.101s. Benchmark not assessed by design.
Manifest: `outputs/stream_integration_20261112/smoke/manifest.json`.
Root source SHA-256: `fe9cb253f3aeeca8b9e48046aeb0f24a25cf6b78d926aeb1c753636e520471a0`.

Root full pytest completed: **493 passed, 1 expected skip**, 105.79 seconds;
session 54440 exited 0. Log: `outputs/stream_integration_20261112/pytest.log`.
Root mesh demo passed (session 59402 exit 0), and git diff --check passed.
Final root verification evidence: `outputs/stream_integration_20261112/checks_manifest.json`. Documentation updated in
`docs/development_feedback.md`, cylinder validation notes and checklist.

Next priorities under this steering: keep routine feedback to targeted checks
and full regression (~2 min), not a new full qualification by default. Implement
strict validated per-case reuse for unchanged cases; complete independent streaming
and cylinder orchestration; profile representative finest-mesh steps and peak RSS
before choosing solver/assembly/thread/parallelism changes. No new 4-hour study
was launched. Acceptance thresholds and CPU numerical method are unchanged.

## GPU performance question

User challenged why GPU/multicore acceleration is not used. Read-only hardware
and library audit completed: AMD Navi 24 gfx1034 with 4,278,190,080 VRAM bytes;
integrated gfx90c with 512 MiB reserved VRAM. Working Torch is CPU-only and PETSc
3.25.6 has no CUDA/HIP/Kokkos, with MUMPS available. Current main SciPy/SuperLU
solver is CPU code. No GPU backend, GPU numerical benchmark or speedup exists.
ROCm management tools are installed, and AMD TheRock's current development table
lists gfx1034 support; do not claim the GPU is categorically unusable. Exact
runtime/library support and useful float64 performance remain untested.

See `docs/acceleration_assessment.md` and ignored capability manifest. Initial
MPI probe hit sandbox restrictions; approved read-only retry with UCX_TLS=self
completed. No dependencies were installed and no numerical source changed.
Acceleration evaluation is now a performance priority: measure CPU parallel and
GPU-capable backend candidates against the preserved reference before another
large qualification run. A CPU baseline is not a mandate for single-threaded
production. Preserve explicit phase-decision requirements for new GPU dependencies.

## Acceleration phase authorized and implemented

Latest user instructions “Begin the next phase” and “Continue” authorize this
phase. Read the current results and commands in `docs/acceleration_assessment.md`.
Installed ROCm 10.1.0 only in ignored `outputs/acceleration_runtime`; kept CPU
environment/system drivers unchanged. SDK checks 19/19 passed with approved
host device access; sandbox hides GPU nodes. Actual rocSPARSE float64 SpMV passed
on both level-16 matrices, but transfer-inclusive SpMV is slower than CPU. This
is GPU arithmetic evidence, not a GPU sparse solver or trajectory acceleration.

72 fresh-process CPU solver trials finished. MUMPS passed all 24 trials, about
2.14–2.36x faster than one-thread SciPy at six requested threads. SuperLU_DIST's
24 default-config trials failed zero pivots and remain recorded. No gate relaxed.
Implemented optional MUMPS backend in root, initial projection/Newton both use it,
resources cleaned and original-system residual checked. SciPy remains default.
Restart contracts record backend/version/thread requests; cylinder CLI and study
support explicit selection. Ten-step MUMPS vs independent FEniCSx comparison passed
(max velocity 9.41e-15, pressure 7.55e-15). Targeted tests: 91 passed in 12.44s.
Artifacts: `outputs/acceleration_phase_20261112/`.

Bounded level-16 startup trajectory comparison (6 steps, 3 repeats each backend)
is now running separately from other CPU/GPU tests. It measures complete local
solve and full process time and checks all velocity, pressure, reactions, force
and pressure-difference histories. No new 4-hour qualification has been launched.
Follow with full pytest and mesh convergence after source is stable. Preserve
previous failed qualification. Further acceleration work requires complete GPU
solver/preconditioner and resident-data design, not replacing SciPy arrays alone.

### Acceleration milestone completed and validated

The six-run level-16 trajectory comparison completed: median local solve
22.834s SciPy vs 13.424s MUMPS (1.70x); median complete process 24.909s vs
15.493s (1.61x). All velocity/pressure/reaction/force histories agree within
1.67e-15 or better. Source unchanged during experiment. This is startup-only,
not a measured full-wake study speedup. Full regression completed **500 passed,
1 expected skip in 121.08s**; mesh convergence 8/16/32 and diff check passed.
Final aggregate evidence: `outputs/acceleration_phase_20261112/validation_manifest.json`.

Current optional MUMPS integration is reviewable and usable via cylinder CLI;
SciPy remains default. No new four-hour study launched. Next bounded validation:
developed-wake continuation/full-run memory for MUMPS; finish independent streaming
and cylinder archive integration; then declare finer mesh/timestep qualification
that addresses the still-failed refinement criteria. GPU follow-up is a complete
resident-data sparse solver/preconditioner, with transfer-inclusive trajectory
verification and peak memory, before adopting it in scientific runs.

## Complete GPU solver experiment: all four objectives

User explicitly requested building the GPU candidate, startup/wake experiments,
six-thread CPU comparison and unchanged acceptance gates. Implemented experimental
`rocm_gmres`: CPU ILUT setup (drop 1e-6/fill 20), vendor rocSPARSE GPU SpMV and
triangular solves, rocBLAS GPU vector operations, host small Hessenberg least
squares. This is a hybrid solver, not all-GPU factorization/assembly. Full vectors
stay resident within the Krylov iteration sequence. Factors reset each timestep;
preconditioning reuse never replaces the current Newton Jacobian.

No new dependencies/system-driver changes. Existing isolated ROCm SDK is used;
PETSc GPU rebuild not necessary for this candidate. Flow wrapper integrates the
backend while SciPy remains default. Restart contracts contain SDK/configuration;
GPU restart checked numerically, not claimed bitwise across devices. Drivers:
`scripts/check_gpu_solver.py`, `scripts/qualify_gpu_solver.py`, existing cylinder
runner/checker with `--linear-backend rocm_gmres`.

Real linear solves on both level-16 fixtures passed <=1e-9 field gates (errors
~1.1e-13/1.3e-13). Real GPU manufactured tests: 4 passed in 2.52s. Independent
10-step cylinder passed (max velocity 9.52e-15, pressure 8.22e-15). Targeted CPU
checks: 54 passed in 6.86s. Immutable archive seeds prepared at steps 0 and 1200,
3 continuation steps each, with original CPU field/reaction frames and checksums.
The seed is a new comparison experiment, not a relabeled qualification checkpoint.

Repeated study running sequentially: 3 repeats x 2 cases x 2 backends =12 fresh
processes, six requested CPU threads for both. Full process wall time and peak
host/device memory recorded; acceptance requires every accuracy check plus >=1.5x
median end-to-end speedup in BOTH cases. Initial pairs are accurate but GPU is
~2x slower. Preserve failures/rejection. Do not claim full-qualification speedup.
Artifacts: `outputs/gpu_solver_phase_20261112/`. Protocol and limitations:
`docs/gpu_solver_trial.md`. No full four-hour qualification launched.

### GPU comparison completed: accurate, rejected on performance

All 12 workers and 6 paired comparisons passed accuracy/source checks. Median
complete-process times: startup MUMPS10.040s vs GPU19.705s; wake MUMPS12.341s vs
GPU25.821s. GPU is 1.96x/2.09x slower; both miss >=1.5 speedup gate. Study finished
in207.54s, exit1 `candidate_rejected_performance` (intentional gate, no solver
crash). Max velocity difference2.45e-15, pressure2.54e-13, drag1.43e-14.

Max host RSS CPU600,797,184 bytes vs GPU1,523,732,480; explicit GPU allocation
peak292,382,404 and sampled device-wide peak708,837,376 bytes (not exact
per-process transient peak). CPU ILUT setup alone cost~8s per3-step window,
plus GPU setup/triangular/Krylov costs. Keep SciPy default and MUMPS available;
do not promote this candidate or claim GPU speedup. Next GPU work requires a
cheaper preconditioner/setup design and another full-cost comparison. Full
pytest and mesh convergence are now running after the completed timed study.

### Final GPU phase validation

Full CPU regression: **506 passed, 5 skipped in 117.72s**. Four skipped GPU tests
were executed separately on real hardware: **4 passed in 2.52s**; the remaining
skip is the missing-FEniCSx-dependency path. Sine convergence8/16/32 passed
(L2rate1.993,H1rate0.997), diff check passed. Final aggregate artifact:
`outputs/gpu_solver_phase_20261112/validation_manifest.json`; its source hash
matches the completed study. All four requested objectives are implemented and
tested; GPU candidate rejected for performance. No work is running in background.
Keep SciPy default/MUMPS optional. GPU backend remains explicitly experimental.
Full physical qualification is still unresolved on spatial/time refinement.
