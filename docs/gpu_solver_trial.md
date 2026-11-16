# Complete GPU solver trial

The user authorized all four objectives: a resident GPU solver candidate,
startup/developed-wake trajectory tests, a fair comparison with six-thread MUMPS,
and unchanged numerical acceptance checks. This trial is separate from full
DFG spatial/time qualification, which remains failed for the previous grid study.

## Implementation and mathematical scope

`linear_backend='rocm_gmres'` opts into an experimental **hybrid** solver. The
finite-element assembly and nonlinear Newton method remain on CPU. CPU SuperLU
ILUT constructs an approximate factorization with `drop_tol=1e-6` and
`fill_factor=20`. GPU rocSPARSE performs sparse multiplication and triangular
preconditioner applications; GPU rocBLAS performs full-vector Krylov operations.
Matrices, factors, basis vectors and solution vectors stay on the GPU throughout
each linear iteration sequence. Only scalar reductions and the small Hessenberg
least-squares system return to the host. This is a complete linear solution and
trajectory path, not an SpMV-only demonstration, and is **not** an all-GPU assembly
or factorization implementation. All CPU work is included in the experiment.

The initial CPU ILUT pilot rejected the cheaper drop=1e-4/fill=10 configuration:
it failed to converge on the startup system after 200 iterations. Its failure is
retained in `outputs/gpu_solver_phase_20261112/cpu_ilu_pilot.json`. The declared
candidate uses the stronger preconditioner on both cases; no acceptance tolerance
was loosened. CPU setup time can dominate and may make this candidate unsuitable.

For each constrained Newton system A*x=b, SuperLU permutations form
B=P_r*A*P_c and c=P_r*b. The preconditioner is M=L*U, approximating B. Restarted
right-preconditioned GMRES solves B*M^-1*z=c, recovers y=M^-1*z and x=P_c*y.
The implementation stores preconditioned basis vectors and uses twice-modified
Gram-Schmidt. It checks the actual residual after each restart/estimated
convergence, not just the projected Hessenberg estimate. The relative Krylov
threshold is 1e-12, restart dimension 30, maximum 120 iterations, with a 60-second
iteration-loop limit. The orchestration additionally limits each worker to 600 s.
Nonfinite values, failed factorization and nonconvergence are explicit failures;
there is no automatic fallback that could mislabel CPU execution as GPU success.

Within a timestep, ILUT factors may precondition subsequent changing Jacobians.
**The actual matrix is rebuilt and applied on every Newton iteration**. Reusing an
approximate preconditioner never substitutes an old matrix for the new equation.
Factors are discarded at every timestep boundary, so a continuation does not
require an unrecorded preconditioner from a prior segment. GPU restart is checked
for numerical agreement; no cross-device bitwise reproducibility is claimed.

SciPy remains the default. MUMPS remains an explicit CPU option. The existing
CPU environment and system GPU drivers were not modified. This implementation
uses the already installed isolated ROCm SDK; a PETSc rebuild was unnecessary
for this candidate. No custom GPU kernels, CUDA or Triton dependencies were added.

## Hardware, precision and memory

Measured hardware: AMD Radeon RX 6500, gfx1034, PCI 0000:03:00.0, reported VRAM
4,278,190,080 bytes; Ryzen 5 5600H, six CPU cores/twelve logical processors.
ROCm SDK/core/libraries 10.1.0 reside under ignored `outputs/acceleration_runtime`.
All values, factors and Krylov operations are float64; CSR indices use int32.

Each solve reports CPU ILUT setup, GPU setup/upload/iteration/download/cleanup,
iteration count, true residual, device identity and explicit device-allocation
peak. It also samples device-wide used VRAM with hipMemGetInfo after allocations,
setup and reductions. This sampled value includes other processes and may miss
transient vendor-library allocation peaks; it is not exact per-process peak VRAM.
Worker manifests report Linux peak host RSS, including imports and FE assembly.
Full-vector transfers occur for each new linear system, not each Krylov iteration.

Vendor interfaces:
[rocSPARSE multiplication and triangular solves](https://rocm.docs.amd.com/projects/rocSPARSE/en/latest/reference/level2.html),
[rocBLAS vector operations](https://rocm.docs.amd.com/projects/rocBLAS/en/latest/reference/level-1.html).

## Declared startup/wake experiment

Immutable seeds come from the completed CPU archive at level 16, dt=.005:

- Startup: step 0, three steps through t=.015.
- Developed wake: step 1200 at t=6, three steps through t=6.015.

The wake seed includes both BDF2 velocity histories and the previous physical
pressure. A new comparison contract explicitly identifies that input. It does
not alter an old checkpoint or claim to resume the qualification study. Original
archive velocity, pressure and nodal-reaction frames are included for additional
checks. Seed checksums, source archive checksum, original source identity and
current implementation identity are preserved. Positive mesh quality and physical
boundary checks use the same package as the CPU baseline.

The study runs three fresh-process repeats per backend per case, sequentially,
with six requested OpenMP/BLAS threads and alternating backend order. Total wall
time includes imports, seed loading, assembly, CPU preconditioning, GPU setup and
transfers, every Newton solve, physical evaluation and output. GPU kernels are
synchronized before results/timings are finalized. Regression tests and other
compute experiments are not run concurrently with these performance trials.

The acceptance gate requires **all** repeats in **both** cases to satisfy:

- Velocity, pressure, nodal reaction and force-history differences <=1e-9.
- Agreement with the archived CPU fields <=1e-9.
- Unchanged Newton/discrete gates: residual <=5e-11, boundary/no-slip <=1e-12,
  weak divergence and flux <=1e-10, momentum balance <=1e-9.
- Identical time grids and unchanged implementation source during the study.
- Median complete-process speedup >=1.5 over six-thread MUMPS in each case.

A numerically accurate but slower candidate is explicitly rejected for adoption.
A missing, failed or duplicate repeat cannot count as a passing study. No result
from these short windows establishes the full wake's spatial/time convergence.

## Commands

The GPU device nodes are hidden inside the execution sandbox. Numerical GPU
commands require approved host device access, without changing permissions.
First set the isolated SDK location and thread limits before Python starts:

```bash
export PHYSGRAPH_ROCM_SITE="$PWD/outputs/acceleration_runtime/lib/python3.12/site-packages"
export OPENBLAS_NUM_THREADS=6 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 UCX_TLS=self
```

Prepare and compare (use new output directories for each experiment):

```bash
.fenicsx-env/bin/python scripts/qualify_gpu_solver.py prepare \
  --trajectory outputs/cylinder_full_qualification_20261112/level16_dt0.005/trajectory.npz \
  --output outputs/my_gpu_seeds --starts 0 1200 --steps 3
.fenicsx-env/bin/python scripts/qualify_gpu_solver.py study \
  --seeds outputs/my_gpu_seeds --output outputs/my_gpu_study --repeats 3
```

Real-device tests and independent short-cylinder check:

```bash
PHYSGRAPH_GPU_TEST=1 .fenicsx-env/bin/python -m pytest -q tests/test_rocm_flow.py
.fenicsx-env/bin/python scripts/check_cylinder_flow.py \
  --linear-backend rocm_gmres --fenicsx --output outputs/my_gpu_smoke
```

Ordinary CPU regression skips explicitly marked GPU tests. Such a skip is not GPU
verification; the real-device command must be executed separately and recorded.

## Initial verification evidence

Full linear solves passed on both archived fine systems (65,061 free unknowns):
startup four Krylov iterations, wake three; maximum absolute CPU-reference errors
1.14e-13 and 1.34e-13. Explicit device-allocation peaks were 255,451,000 and
292,382,404 bytes; sampled device-wide peaks 671,088,640 and 708,837,376 bytes.
Cold first-solve/context cost is included and is not a steady-state timing claim.

All four real-device manufactured-flow tests passed in 2.52 s: BE/BDF2, nonzero
Dirichlet lifting, closed/traction pressure constraints, analytical field errors,
CPU reaction parity and resumed-segment agreement. The independent ten-step
cylinder comparison passed: maximum velocity difference 9.52e-15, pressure
8.22e-15, with every physical step and the initial projection checked. Targeted
CPU seed/Krylov/checkpoint/backend tests: 54 passed in 6.86 s.

Artifacts are in ignored `outputs/gpu_solver_phase_20261112/`; the repeated study
and final regression outcomes are recorded below when complete.

## Completed comparison and adoption decision

**Decision: reject this candidate for performance; retain the CPU baseline and
MUMPS development option.** All 12 workers and all six paired comparisons passed
the numerical gates. Source identity remained unchanged throughout the study.
The orchestrator exited 1 with `candidate_rejected_performance`, an intentional
acceptance result, not a crashed or unconverged simulation.

| Three-step window, median of three repeats | MUMPS, 6 threads | Hybrid GPU GMRES, 6 CPU threads | GPU/CPU elapsed ratio |
|---|---:|---:|---:|
| Startup, complete process | 10.040 s | 19.705 s | 1.96x slower |
| Developed wake, complete process | 12.341 s | 25.821 s | 2.09x slower |
| Startup, local solver only | 8.254 s | 18.297 s | 2.22x slower |
| Developed wake, local solver only | 10.424 s | 23.996 s | 2.30x slower |

The measured speedup values are 0.510 and 0.478, below the declared minimum 1.5.
The entire 12-run study completed in **207.54 seconds**. These are bounded-window
results, not timings or qualification of the full physical [0,8] trajectory.

Maximum CPU/GPU differences across all repeats/windows:

| Quantity | Maximum absolute difference |
|---|---:|
| Velocity | 2.45e-15 |
| Pressure | 2.54e-13 |
| Nodal reactions | 2.41e-15 |
| Drag coefficient | 1.43e-14 |
| Lift coefficient | 7.64e-15 |
| Pressure difference | 9.77e-15 |
| Surface drag coefficient | 1.54e-14 |

Both backends also matched the original archived field/reaction frames within
1e-9; the largest archived pressure discrepancy was 2.25e-13. This validates the
common BDF2 seed and the fresh integrator path against the earlier CPU evidence.

Peak host RSS across trajectory workers: MUMPS 600,797,184 bytes; GPU candidate
1,523,732,480 bytes. GPU explicit allocation peak 292,382,404 bytes; sampled
device-wide used-memory peak 708,837,376 bytes. The memory-measurement limitations
above apply; full longer-run residency and different meshes were not tested.

The first startup repeat spent 8.16 s building three CPU ILUT preconditioners and
6.22 s in GPU setup/transfers/iterations/cleanup. In the first wake repeat those
costs were 7.62 s and 10.91 s. These are individual-repeat breakdowns, not medians.
The startup used six linear solves/23 Krylov iterations; the wake used nine/54.
This explains why faster resident multiplication did not produce a faster solver.

The next GPU candidate would need a substantially cheaper preconditioner/setup
path, likely exploiting the velocity/pressure block structure and more GPU-native
setup. It must undergo the same complete-trajectory comparison; changing isolated
kernel timings or lowering precision does not satisfy the gate. This result does
not prove every GPU algorithm or larger GPU is slower. No full four-hour study was
launched, and the original spatial/time refinement failures remain unresolved.

Primary artifacts:
`outputs/gpu_solver_phase_20261112/study/manifest.json` plus per-worker manifests,
trajectory archives and logs. The manifest links seed/source identities and every
worker's checksummed report, timings, residuals and memory records.

## Final regression and artifact audit

Full CPU suite: **506 passed, 5 skipped in 117.72 s**. Four skips are the explicit
GPU tests, which were executed separately on the real device and **all four
passed in 2.52 s**; the other skip exercises missing FEniCSx dependencies in an
environment where they are installed. The sine mesh convergence demo on 8/16/32
passed, with finest L2/H1 rates 1.993/0.997. `git diff --check` passed.

`outputs/gpu_solver_phase_20261112/validation_manifest.json` records all four
objectives as implemented/tested, with the candidate rejected on performance.
It links study, seeds, linear solves, independent verification and test logs by
checksum. Its source hash matches the completed repeated study. The exploratory
CPU ILUT pilot has a separately identified retrospective provenance manifest;
its recording source hash is not presented as a historical launch snapshot.
