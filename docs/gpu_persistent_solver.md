# Persistent GPU solver development — 13 November 2026

User instruction: continue GPU development, but run final qualification **only**
when the GPU demonstrates considerable speedup over CPU. The existing proposed
minimum is 1.5x complete-process speedup over six-thread MUMPS in both startup and
wake windows. Three-step tests are screening only. Longer windows, independent
verification, and an equivalent CPU reuse control are required before a full run.
No final qualification is launched by the development benchmark scripts.

## Profiling and rejected designs

Synchronized operation profiling is recorded under
`outputs/gpu_persistent_20261113/profile`. It isolates CSR upload/analysis,
preconditioner application, multiplication, reductions and vector updates. Nested
CSR/upload timings are not additive; instrumentation is not acceptance timing.
The profile identified sparse analysis and triangular preconditioner application
as substantial costs, in addition to CPU ILUT construction. The machine initially
reported AC disconnected. Those timings cannot be compared as controlled
performance results with the earlier AC runs. The user connected AC, which was
verified through `/sys/class/power_supply/ACAD/online` before subsequent trials.
No power-governor or driver setting was changed.

Two CPU block-preconditioner pilots were evaluated before GPU integration:

- Velocity/pressure smoothed aggregation: startup did not meet Krylov convergence
  within 160 iterations; the wake hierarchy rejected a nonpositive coarse diagonal.
- Symmetric velocity multigrid plus pressure-only LU: startup converged in 140
  iterations; wake did not meet the requested convergence within the budget.

These candidates are **not** available as trajectory backends. Failures remain in
`amg_cpu_pilot` and `block_cpu_pilot_v2`. PyAMG 5.4.1 was installed without dependencies
under ignored `outputs/gpu_persistent_20261113/python_packages`; the CPU environment
was not modified. `PHYSGRAPH_AMG_SITE` selects this optional research dependency.
No threshold was loosened to make either pilot pass.

## Selected persistent candidate

`rocm_cached` uses float64, restarted right-preconditioned GMRES and a CPU SuperLU
factorization as a reusable preconditioner. Unlike the initial ILUT candidate,
the factorization is exact for its setup matrix. It is **not** used as the direct
solution of later changing equations: the current Jacobian is uploaded and used
in every matrix-vector product and residual check.

GPU factor storage, triangular analysis, handles and Krylov workspaces persist
across solves. Current CSR numerical values are updated every solve; indices are
updated when the pattern changes. Buffer capacity grows when required. The
preconditioner refreshes at global 32-step boundaries, after more than 12 Krylov
iterations, or once after nonconvergence. A retry rebuilds the preconditioner and
still solves on GPU; it never silently switches to a CPU solution. Retry failures,
setup counts, update/iteration costs and memory are recorded.

The initial projection has its own cache lifetime. Step 1 resets the cache.
A restarted trajectory rebuilds its preconditioner; full physical histories and
boundary values are preserved. Restart agreement is numerical, not a promise of
bitwise identity across different preconditioner histories. Runtime/source/backend
contracts still reject incompatible saved states.

`cpu_cached` is the control: the same fixed-step refresh policy, permutations,
right-preconditioned Krylov algorithm and convergence tolerances, with CPU sparse
factor application. This prevents attributing a reuse improvement solely to GPU
hardware. MUMPS requests six threads; cached CPU/GPU candidates request one CPU
thread to avoid multithreading overhead in the many small host reductions.
SciPy remains the default and the original `rocm_gmres` candidate remains available
for reproducing the earlier rejected design.

## Validation and commands

Targeted CPU tests verify changing matrices/right-hand sides, workspace reset,
refresh boundaries, nonzero boundary values and restart agreement. The real-GPU
tests now run both GPU backends across BE/BDF2 and mean/traction constraints.
Eight CPU targeted tests and eight real-GPU tests passed before timing trials.
Full regression and convergence evidence is recorded after development stabilizes.

```bash
export PHYSGRAPH_ROCM_SITE="$PWD/outputs/acceleration_runtime/lib/python3.12/site-packages"
.fenicsx-env/bin/python scripts/qualify_gpu_solver.py study \
  --seeds outputs/gpu_solver_phase_20261112/seeds \
  --output outputs/my_cached_gpu_vs_mumps \
  --candidate rocm_cached --baseline mumps
.fenicsx-env/bin/python scripts/qualify_gpu_solver.py study \
  --seeds outputs/gpu_solver_phase_20261112/seeds \
  --output outputs/my_cached_gpu_vs_cpu \
  --candidate rocm_cached --baseline cpu_cached
```

GPU commands need approved host device access because the sandbox hides device
nodes. Trials run sequentially; source changes during a study invalidate it.
Use new output directories and preserve rejected results. Report the RX 6500 /
gfx1034 identity, 4,278,190,080-byte reported VRAM, float64 precision, process wall
time, explicit allocated GPU peak, sampled device-wide memory and host peak RSS.
Sampled GPU use is not an exact per-process transient allocation peak.

## AC-powered screening results

The sequential 12-process comparison completed with every numerical and source
check passing. All timings include imports, input/output, assembly, CPU setup,
GPU transfers, Newton iterations and evaluation. Medians of three repeats:

| Three-step window | Six-thread MUMPS | Persistent GPU | CPU/GPU speed ratio |
| --- | ---: | ---: | ---: |
| Startup | 10.084 s | 13.642 s | 0.739x |
| Developed wake | 12.790 s | 19.652 s | 0.651x |

Evidence: `outputs/gpu_persistent_20261113/study_mumps/manifest.json`.
Status is `candidate_rejected_performance`, not a numerical failure. Maximum
paired pressure difference was 1.93e-13. The candidate remains experimental.
No longer-window acceptance study or full qualification is justified by these
results. Older battery-powered diagnostic timings are not acceptance baselines.
Peak explicit GPU allocations were 348,628,120 bytes, with a sampled device-wide
peak of 763,363,328 bytes. Peak host RSS was 1,534,889,984 bytes for the GPU
candidate versus 612,384,768 bytes for MUMPS. A live sample during the subsequent
control study recorded AC online, 99% GPU activity and a 2,610 MHz shader clock
(`power_sample_3.json`). This is a point sample, not an average utilization claim.

## Vendor GPU numerical refactorization pilot

An isolated pilot uses AMD's documented
[rocSOLVER sparse refactorization API](https://rocmdocs.amd.com/projects/rocSOLVER/en/latest/reference/refact.html).
Initial CPU SuperLU selects pivots and factor sparsity. GPU numerical LU and
triangular solution then operate on changed values of the same sparse structure.
The pilot rejects pattern changes; it is deliberately not a trajectory backend.
No custom GPU kernels or new GPU dependencies were introduced.

`rocm_refactor.py` preserves stored zero fill entries in the bundled factors and
checks the original, unpermuted system. Up to three corrections solve the true
residual with the GPU factors. No CPU solution fallback is permitted. Fixed pivots
can lose accuracy, so successful small perturbations do not establish stability
on arbitrary later Newton matrices. The first startup attempt without correction
failed the 2e-12 relative residual threshold and remains preserved separately.

After correction, the startup and wake fixtures and two deterministic nonuniform
value perturbations each passed. GPU solve maximum differences from fresh CPU LU
were 1.78e-13 (startup) and 4.20e-14 (wake). Median warmed transfer/refactor/solve/
check time was 1.528 s versus 1.513 s for fresh CPU LU in startup, and 1.535 s
versus 1.153 s in wake. These CPU controls use one thread and are not MUMPS.
GPU figures exclude initial CPU factorization and GPU analysis, which are recorded
separately. Even with that favorable exclusion, the pilot does not show useful
acceleration, so it was not integrated into flow trajectories.

Artifacts: `refactor_startup` (preserved initial rejection),
`refactor_startup_v2`, and `refactor_wake` below
`outputs/gpu_persistent_20261113/`. Each pilot has a two-minute external execution
limit, fixture checksums, source fingerprint, float64 precision, errors and GPU
memory measurements. This is linear-system evidence, not physical qualification.

```bash
.fenicsx-env/bin/python scripts/pilot_gpu_refactor.py \
  --fixtures outputs/acceleration_phase_20261112/fixtures \
  --output outputs/my_refactor_wake --case step01201
```
