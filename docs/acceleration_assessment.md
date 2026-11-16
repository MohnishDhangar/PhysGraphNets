# Acceleration assessment — 12 November 2026

The CPU reference is implemented and profiled. A GPU numerical backend has not
been implemented or benchmarked. Keeping the CPU reference does not require all
future production runs to remain single-threaded.

## Observed local hardware and installed software

Read-only checks (`lspci`, `rocm-smi`, sysfs and runtime library queries) found:

- Discrete AMD Navi 24, PCI device 1002:743f, architecture gfx1034; reported VRAM
  4,278,190,080 bytes (approximately 4 GiB). Marketing model is not resolved by
  these checks; PCI naming includes RX 6400/6500 XT/6500M variants.
- Integrated AMD Cezanne/gfx90c; 536,870,912 bytes of reported reserved VRAM.
- Working Torch 2.10.0+cpu: CUDA/HIP builds absent; GPU availability false.
- Working PETSc 3.25.6: CUDA=false, HIP=false, Kokkos=false, MUMPS=true.
- Current main flow solver calls SciPy spsolve/SuperLU with CPU arrays. Its
  sparse linear solves dominated the previous short profile. GPU-enabled Torch
  alone would not change this FEM execution path.

The first combined capability probe hit sandbox MPI restrictions; a subsequent
approved read-only probe with UCX_TLS=self completed. GPU management output had
limited product-name access, so architecture/VRAM were corroborated by PCI/sysfs.
No GPU numerical calculation, timing, peak-memory measurement, library install or
source-backend change was performed. Capability evidence is saved in
`outputs/acceleration_assessment_20261112/capabilities.json`.

## Feasible direction and evidence required

[PETSc's GPU documentation](https://petsc.org/release/overview/gpu_roadmap/)
describes HIP/ROCm matrix/vector support for AMD GPUs. The working PETSc build
has none of those GPU capabilities. A suitable GPU-capable solver must accompany
GPU matrix storage; changing an array/device flag is not a completed port of the
current nonlinear sparse solver.

[AMD's TheRock development support table](https://github.com/ROCm/TheRock/blob/main/SUPPORTED_GPUS.md)
lists gfx1034 with build, sanity-test and release-readiness indicators. This is a
development support table, not proof that every installed ROCm library works on
this machine. The [ROCm compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)
and an actual local compute test must guide backend selection. Do not dismiss
the local GPU as unusable solely because an older release did not list it.

The next acceleration decision should compare measured multicore CPU and GPU
solver paths on representative sparse systems and short trajectories. Preserve
the verified CPU reference, float64 numerical contract, existing error thresholds
and independent verification. Report hardware, VRAM, transfer-inclusive timing,
precision, residual/field/load differences and peak memory. Four GiB is an input
to memory testing, not evidence that the workload definitely fits or fails.

AGENTS.md requires a phase decision before new CUDA/Triton dependencies and
retaining a strict CPU baseline. It does not permanently prohibit acceleration.
The current question authorizes capability investigation; no new dependency or
custom kernel was introduced. A measured backend trial should precede another
large qualification run or any promise of GPU speedup.

## Authorized acceleration phase and measured results

The user's subsequent “Begin the next phase” and “Continue” authorize isolated
acceleration dependencies and backend evaluation. The earlier capability-only
paragraphs describe the preceding audit, not the current authorization. No system
driver or CPU environment dependency was replaced. The reference remains SciPy.

An isolated ROCm 10.1.0 SDK with the gfx1034 device package was installed under
ignored `outputs/acceleration_runtime`. Outside the sandbox, all 19 SDK installation
checks passed. The runtime reports **AMD Radeon RX 6500**, PCI 0000:03:00.0,
gfx1034, and **4,278,190,080 bytes VRAM**. Sandbox GPU device nodes are hidden;
GPU execution needs approved host device access. Do not modify device permissions.

### Sparse CPU solver comparison

`outputs/acceleration_phase_20261112/cpu_study/manifest.json` records 72 fresh-process
trials, three repeats of SciPy, MUMPS and SuperLU_DIST at 1/2/4/6 requested threads
on startup and developed-wake Newton systems from the archived level-16 trajectory.
Both systems have 65,061 free unknowns. Every trial includes matrix conversion,
factorization, solution and result copy; imports and fixture reads are excluded.
Thread counts are requests through OpenMP/BLAS environment variables, not a claim
that all solver operations use that many cores. CPU: Ryzen 5 5600H, 6 cores/12
logical processors. Peak process RSS is preserved per trial.

| System | SciPy, 1 thread | MUMPS, 6 threads | Linear-solve ratio |
|---|---:|---:|---:|
| Startup, step 1 | 1.525 s | 0.646 s | 2.36x |
| Developed wake, step 1201 | 1.294 s | 0.605 s | 2.14x |

All 24 MUMPS trials met residual <=1e-10 and absolute reference difference <=1e-9.
All 24 default SuperLU_DIST trials failed with zero pivots. This rejects the tested
configuration, not every possible SuperLU_DIST ordering/pivoting configuration.
No shifted matrix or relaxed tolerance was used to make it pass. SciPy thread
increases did not provide a useful improvement. These timings do not predict a
2x complete qualification speedup: assembly, output and independent verification
also take time. Small SDK installation checks overlapped part of the first repeat;
the following complete-trajectory comparison runs separately from other tests.

### GPU arithmetic evidence

`outputs/acceleration_phase_20261112/gpu_probe/manifest.json` records 30 synchronized
repetitions per matrix using vendor **rocSPARSE dcsrmv**, float64 values and int32
CSR indices. This operation computes A*x; it does **not** solve A*x=b.
No custom GPU kernels were implemented. API references:
[rocSPARSE level-2 operations](https://rocm.docs.amd.com/projects/rocSPARSE/en/latest/reference/level2.html),
[ROCm Python packaging](https://github.com/ROCm/TheRock/blob/main/docs/packaging/python_packaging.md).

| System | CPU SpMV | GPU resident SpMV | GPU including transfers |
|---|---:|---:|---:|
| Startup | 1.144 ms | 0.266 ms | 4.820 ms |
| Developed wake | 1.753 ms | 0.383 ms | 7.096 ms |

Maximum absolute errors were 1.02e-18 and 3.26e-19. Resident timings exclude
allocation and uploads; transfer-inclusive timings upload the complete CSR and
vectors and download the result each repetition. Each GPU call is synchronized.
The first cold initialization/allocation/upload/compute/download took 0.838 s;
the second matrix reused an already initialized process, so its 0.009 s setup
must not be described as an independent cold-start measurement.

Explicit device allocation peaks were 15,553,468 and 23,280,088 bytes. Sampled
**device-wide** used-memory peaks were 239,075,328 and 245,366,784 bytes, using
hipMemGetInfo after allocations and synchronized iterations. They include runtime
and other-process usage, may miss transient library peaks, and are not exact
per-process peaks. This demonstrates SpMV capacity only, not sparse LU capacity.
The complete GPU solver remains a future task: keep matrices/vectors resident,
choose and verify a suitable sparse solver/preconditioner, then measure full
Newton/trajectory performance, transfers, numerical errors and memory.

### Available CPU implementation

`solve_navier_stokes(..., linear_backend='mumps')` now selects optional real-float64
PETSc/MUMPS. SciPy remains the default and is available without PETSc. Every MUMPS
solve checks convergence, finite values and the original-system residual, and
releases PETSc objects. The initial projection uses the selected backend too.
Checkpoint contracts include backend, PETSc version and requested thread limits;
changing these rejects continuation. Source guards also reject old-source
checkpoints; do not rewrite old manifests to bypass them.

The cylinder runner and complete study accept `linear_backend`; the study JSON
may explicitly add `"linear_backend": "mumps"`. The existing qualification config
continues to use the default SciPy reference. Example bounded independent check:

```bash
OPENBLAS_NUM_THREADS=6 OMP_NUM_THREADS=6 UCX_TLS=self \
  .fenicsx-env/bin/python scripts/check_cylinder_flow.py \
  --linear-backend mumps --fenicsx --output outputs/my_mumps_check
```

Set thread limits **before** Python starts. The installed `.fenicsx-env` already
provides MUMPS; the ordinary CPU package has no new mandatory dependency.
The ten-step cylinder check passed all discrete checks and independent FEniCSx
comparison: maximum velocity difference 9.41e-15, pressure 7.55e-15. The benchmark
is deliberately not assessed for this short run. Four additional manufactured
trajectory tests cover BE/BDF2, nonzero Dirichlet values, mean/traction pressure
constraints, reaction parity and exact MUMPS restart; backend-switch resumes fail.

Full-trajectory timing/parity and regression evidence are recorded separately
below. The prior full wake qualification still fails spatial/temporal refinement;
acceleration does not change that scientific result or any acceptance threshold.

### Complete bounded trajectory comparison

`outputs/acceleration_phase_20261112/trajectory_comparison/manifest.json` completed
all six fresh-process runs: three repeats per backend, level 16, dt=.005,
six steps to t=.03. Candidate order alternated. Source hashes remained unchanged,
and every discrete and cross-backend parity check passed.

| Median, three repeats | SciPy / 1 requested thread | MUMPS / 6 requested threads |
|---|---:|---:|
| Complete local solve, including assembly/projection/Newton | 22.834 s | 13.424 s |
| Entire process, including startup/archive/evaluation | 24.909 s | 15.493 s |

That is **1.70x faster local solution** and **1.61x faster entire process** for
this bounded startup case. Maximum differences across all repeats: velocity
2.31e-16, pressure 1.67e-15, nodal reactions 1.13e-17, drag 7.78e-16, lift
3.40e-16 and pressure difference 4.45e-16. Surface-force histories also passed.
This is not a measured speedup for the full eight-second wake trajectory or its
independent solve. Do not multiply the old four-hour duration by these ratios to
promise a completion time. The old failed study remains immutable evidence.

CPU linear-trial peak process RSS, including imports/fixtures: SciPy 321.23 MiB;
MUMPS 308.04 MiB. Maximum MUMPS relative residual 1.14e-16 and reference solution
difference 9.41e-14. These memory numbers apply to the isolated matrix trials;
full trajectory histories and FE assembly have additional memory costs.

Reproduce the bounded comparison (existing output directories must be preserved):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 UCX_TLS=self \
  .fenicsx-env/bin/python scripts/compare_flow_backends.py \
  --output outputs/my_backend_comparison --level 16 --steps 6 --repeats 3
```

The orchestrator starts separate processes with each candidate's declared thread
limits. Run timing experiments separately from regression tests or other compute
jobs. Next performance decisions: evaluate developed-wake trajectory continuation
and longer-run memory with MUMPS; then implement/verify a resident GPU solver if
its complete Newton cost is competitive. Continue finer spatial/time refinement
before deployment; a faster solver cannot make an under-resolved case qualified.

### Final regression evidence

Full `.fenicsx-env/bin/python -m pytest -q`: **500 passed, 1 expected skip** in
121.08 s. The skip specifically exercises missing FEniCSx dependencies and is
inapplicable in this environment. The sine mesh convergence demo (8/16/32) passed,
with finest L2/H1 rates approximately 1.99/1.00. `git diff --check` passed.
`outputs/acceleration_phase_20261112/validation_manifest.json` links the timing,
GPU, independent and regression artifacts by checksum, with Python/code identity.
No full qualification speedup, full GPU solver, or deployment qualification is
claimed by these checks.

## Subsequent complete GPU candidate

The user's next instruction authorized implementation of all four full-solver
objectives. A hybrid GPU GMRES backend, startup/wake seed fixtures, repeated
six-thread-MUMPS comparison, and unchanged numerical/performance gates are now
implemented. Earlier statements here that no complete trajectory GPU path exists
refer to the preceding SpMV-only milestone. See [complete GPU solver trial](gpu_solver_trial.md)
for the algorithm, CPU setup retained in the candidate, memory methodology,
real-device tests and the measured adoption decision.
