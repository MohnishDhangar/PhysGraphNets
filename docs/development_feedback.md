# Fast development feedback and full numerical qualification

The completed cylinder study took 15,145 seconds (4 hours 12 minutes). This is a
scientific refinement and independent-verification workload, not the ordinary
software regression suite. Use short checks while developing; perform the full
study when a numerical milestone needs qualification.

## Recorded full-study work

| Work | Steps | Recorded time |
| --- | ---: | ---: |
| Mesh level 4, dt=.005 | 1,600 | 4.22 min |
| Mesh level 8, dt=.005 | 1,600 | 19.08 min |
| Mesh level 16, dt=.005, plus full independent comparison | 1,600 + 1,600 independent | 149.76 min |
| Mesh level 16, dt=.02 | 400 | 29.35 min |
| Mesh level 16, dt=.01 | 800 | 49.98 min |

These are measured end-to-end per-case durations, including evaluation and I/O.
The old records do not separately time the independent solver, so no precise
split is claimed for the 149.76-minute case. Newly instrumented cases record
meshing, local solve, archive write, force/pressure evaluation, and independent
comparison durations in `metrics.timings_seconds`.

There were 6,000 local time steps and 1,600 independent time steps. Local solves
used 14,035 Newton iterations; the independent solve used 3,951. At mesh 16,
29,944 quadratic velocity nodes and 7,651 pressure nodes form 67,539 unknowns
before Dirichlet elimination. Each local Newton correction assembles the
advection Jacobian and calls a fresh sparse direct solve. Each timestep depends
on the preceding state. The five trajectories were scheduled sequentially.

A prior ten-step mesh-8 profile measured 8.679 seconds inside the local solver:
sparse direct solves consumed 4.724 seconds (54.4%), and convection/Jacobian
work consumed 2.291 seconds (26.4%). This identifies optimization candidates;
it is not a measured breakdown of the entire four-hour finest-mesh study.
Alternative sparse orderings previously failed to show a consistent benefit.

The run used CPU float64 with OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1 on
an AMD Ryzen 5 5600H host. Changing thread counts or running cases concurrently
needs measured timing and peak-memory evidence. Current available RAM is limited;
more parallel work is not automatically faster. No GPU dependency is required
for the workflow improvements below.

## Development loop

1. Run tests for the affected numerical or I/O module.
2. Run the short cylinder check after relevant flow changes:

   ```bash
   env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 UCX_TLS=self \
     .fenicsx-env/bin/python scripts/check_cylinder_flow.py \
     --output outputs/cylinder_smoke_NEW --fenicsx
   ```

   The command runs mesh 2, dt=.02, final time .2: ten local and ten independently
   evolved steps. A measured execution took 2.893 seconds inside the experiment,
   excluding interpreter startup: local solve .526 seconds, independent comparison
   2.101 seconds. All discrete and independent checks passed. The published wake
   benchmark is explicitly `not_assessed`; a short run cannot qualify it. Cached
   FEniCSx compilation and machine load affect this time.

3. For numerical changes, run the complete `python -m pytest -q` suite and the
   mesh convergence demo. The preceding staging suite passed 493 tests with one
   expected skip in 117.45 seconds. This is the normal broader development check.
4. Run a targeted full-duration case when changing wake physics or resolution.
5. Run the complete mesh/time/independent study for numerical qualification.

Use fresh output directories to preserve experiment evidence. See
[checkpoint/restart](flow_restart.md) and [segmented output](flow_streaming.md)
for interruption recovery. They avoid losing completed work and bound field
storage memory; they do not make each linear solve intrinsically faster.

## Further speed work

- Add validated per-case reuse to the study runner so exact matching completed
  cases are not recomputed. Cache identity must include scientific source,
  physical configuration, solver settings, environment and artifact integrity.
- Finish streamed independent comparison and cylinder orchestration before
  extending much larger qualification studies.
- Profile representative finest-mesh startup and developed-wake steps, including
  wall time and peak RSS, before changing assembly, factorization or scheduling.
- Evaluate invariant assembly reuse and solver backends/preconditioners against
  the verified CPU direct-solve reference. Do not reuse an old nonlinear matrix
  as if it were unchanged. Acceptance tolerances remain fixed.

Recorded evidence: `outputs/cylinder_full_qualification_20261112/manifest.json`,
`outputs/cylinder_qualification_20261112/profile_level8.prof`, and
`outputs/stream_integration_20261112/smoke/manifest.json`.

Optional acceleration is now available: cylinder commands accept
`--linear-backend mumps`, using the existing PETSc/MUMPS installation and thread
limits set before Python starts. The ordinary SciPy CPU reference stays the
default. The measured level-16 six-step startup comparison was 1.70x faster for
the local solve and 1.61x for the whole process, with all field/load differences
below 1e-9. See [acceleration evidence](acceleration_assessment.md) for hardware,
repeats, precision, memory and limits. The subsequent [complete GPU solver trial](gpu_solver_trial.md)
implemented a hybrid GPU GMRES trajectory path. It passed accuracy checks but
was about twice as slow as MUMPS, so it remains experimental. CPU ILUT setup and
GPU triangular/Krylov/setup costs are included in that decision.
