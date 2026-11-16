# Cylinder-flow development evidence — 12 November 2026

The DFG 2D-3 implementation and complete-study tooling are implemented. The full
study completed with failed spatial and temporal acceptance; scientific
qualification remains open. See [mathematical contract, signed reference audit,
acceptance gates and reproduction](cylinder_flow.md).

## Executed regression

- Installed CPU Torch/FEniCSx environment, Python 3.12.13:
  `python -m pytest -q`: **408 passed, 1 expected skip**, 127.02 seconds.
  The skipped test exercises the missing-dependency path.
- Lightweight NumPy/SciPy environment: **343 passed, 36 optional-dependency
  skips**, 109.40 seconds. No Torch or FEniCSx dependency was added.
- Offline wheel build passed; isolated imports from the wheel and a reference
  status smoke check passed from `/tmp` with Python isolation enabled in both
  environments.
- Pickle-free archive readback passed for level-4/8 dt=0.005 trajectories:
  each stores 1,601 velocity times including zero and 1,600 pressure times,
  with P2 connectivity and all five boundary tags. Readback checks the data
  contract, not field accuracy.
- Original Poisson demo at n=8/16/32 completed. Relative L2 errors:
  0.0422122429 / 0.0107514232 / 0.0027006552; relative H1 seminorm errors:
  0.1943775294 / 0.0979257566 / 0.0490561759.
- New tests verify signed raw pressure, exact linear history interpolation,
  known history offsets, invalid time grids/protocols, failure accounting,
  refusal to qualify without independent verification, and actual short-run
  trajectory/CSV export. Existing cylinder tests cover nonzero inlet/no-slip,
  triangle orientation/area, probes, quadrature, Laplacian manufactured patches
  and independent initial projection/fields/reactions.

Regression source SHA-256:
`8aa037e8066008c42e43586b7294a5d96fb5dc390082fcc6fa5a53fbeb5ae774`.
Revision: `fdf0d8476456e5340a71d7c0fef02a5ddf568273`, with local changes.

## Complete development trajectories

The original diagnostic failure was already corrected in the working code when
this increment began. Complete runs now reach the end of [0,8].

| Mesh level | dt | Maximum drag | Maximum lift | Time of maximum lift | Final signed pressure difference |
| --- | --- | --- | --- | --- | --- |
| 4 | 0.04 | 2.71180645 | 0.02791382 | 4.08 | -0.13080130 |
| 8 | 0.01 | 2.93881191 | 0.41659846 | 5.78 | -0.10022393 |
| 4 | 0.005 | 2.71181806 | 0.07219453 | 6.53 | -0.11336662 |
| 8 | 0.005 | 2.93886967 | 0.43995226 | 5.75 | -0.10533726 |

All four pass their discrete checks and fail published-quantity acceptance.
The level-8 dt=0.005 full independent comparison also passes. The level-8 dt=0.01
case has 0.168% drag error, 12.75% lift error and 0.08719-second lift peak-time
error. Against the corrected signed raw reference its final pressure error is
10.15%. The two runs change both mesh and timestep, so this is **not** an isolated
spatial or temporal convergence study.

At level 4, reducing dt from 0.04 to 0.005 leaves maximum drag essentially
unchanged and lift far below the reference. The fine-timestep pressure error is
1.64%, while lift error remains 84.88%; passing pressure alone is insufficient.
This single-mesh budget comparison does not establish a temporal convergence rate.

Reducing level-8 dt to 0.005 improves lift but does not meet acceptance:
drag error 0.1663%, lift error 7.8610%, final pressure error 5.5622%, and lift
peak-time error 0.0571875 seconds. These values were recomputed from the
pickle-free exported reaction history and pressure field, not inferred from a
plot. Neither a two-budget comparison nor independent agreement substitutes
for the declared three-level space/time study.

## Complete independent trajectory

The level-8 dt=0.005 case completed **1,600 independently evolved FEniCSx
timesteps plus its own initial constrained projection**. All comparisons pass
the fixed 1e-9 maximum absolute tolerance:

| Quantity | Maximum absolute difference |
| --- | --- |
| Velocity coefficients | 1.13213e-10 |
| Pressure coefficients | 1.11381e-10 |
| Boundary flux | 7.21645e-16 |
| Boundary conormal load | 9.27640e-12 |
| Total Dirichlet reaction | 1.64778e-12 |
| Cylinder reaction | 2.07542e-13 |

Python 3.12.13, DOLFINx 0.10.0, PETSc 3.25.6/MUMPS, CPU float64.
The mesh has 2,240 pressure vertices, 8,628 velocity nodes and 4,148 triangles.
Nonzero inlet and cylinder no-slip errors are zero. Maximum weak-divergence
residual is 1.22943e-16, integrated net flux 3.33067e-16 and algebraic momentum
imbalance 1.45430e-12. Maximum normalized Newton residual is 9.74917e-13.
These are discrete checks, not continuum solution errors.

Full-history surface/reaction disagreement is 1.9414% of the reference drag
peak and 0.8571% of the reference lift peak, both below the declared 5% limit.
The 1,991.89-second execution included solving, archive I/O, diagnostics and
independent evolution, and overlapped other checks. It is not an isolated
performance benchmark. The launch/final source fingerprint matched exactly.

Independent agreement supports consistency of these implementations on the
same mesh/time discretization. It does not overturn the failed published
benchmark quantities or qualify a continuum cylinder wake.

At fixed dt=0.005, level-4 to level-8 lift histories differ by 32.98% RMS of
the reference lift scale. At fixed level 8, dt=0.01 to 0.005 lift histories
differ by 13.57% RMS. Both exceed the declared 5% history criterion; neither
spatial nor temporal stability has been established. Pairwise comparisons are
recorded with input-manifest checksums, without claiming three-level rates.

The first two development manifests retain their original unsigned-pressure reference;
they are not rewritten. Their `launch_manifest.json` files preserve the source
identity before new study modules were added during execution. They predate the
new source-unchanged guard and are development evidence, not final qualification.

The official raw pressure ZIP was downloaded and its final level-6 front/rear
values checked directly. The signed difference is -0.11154138872, whereas the
HTML table lists its positive magnitude. The archive checksum and its slightly
earlier final timestamp are recorded in the evaluator and formulation document.
Neither the 3% quantity nor 0.05-second peak-time tolerances were relaxed.

## CPU diagnostic

A ten-step level-8 cProfile run identified sparse direct factorization as the
largest cumulative cost (4.72 of 11.70 profiled seconds); convection took 2.29
seconds. Separate three-repeat synthetic Newton-matrix ordering checks at levels
8/16 did not establish a useful consistent ordering improvement. The default
SciPy linear solver and numerical method were retained. These diagnostics are
not a controlled end-to-end solver speed claim.

## Artifacts and remaining gate

Artifacts are under ignored `outputs/cylinder_qualification_20261112/`:
development trajectories, original/launch manifests, force CSV, CPU profiles,
`ordering_profile.json`, `development_assessment.json`, archive/readback/build
manifests and regression outputs. The final fine run is
`level8_dt0005_independent/manifest.json`.

The declared qualification requires three meshes at fixed finest dt, three
timesteps on the finest mesh, full independent finest-trajectory comparison and
surface/reaction agreement. The configured level-4/8/16, dt=0.02/0.01/0.005 study
completed in 15,145.26 seconds. All five saved trajectories passed the post-run
artifact audit. Discrete checks, finest published benchmark, surface/reaction
agreement and all 1,600 independent finest steps passed. Finest spatial and
temporal refinement gates failed. See
`outputs/cylinder_full_qualification_20261112/assessment.md` and
`artifact_audit.json` for final evidence. Earlier sections above describe the
preserved development studies preceding that run. No qualified cylinder wake,
learned fluid model, particle-around-obstacle implementation or deployment
readiness is claimed.

For normal development use the [fast feedback workflow](development_feedback.md).
A short cylinder check with independent comparison completed in 2.893 seconds;
it explicitly leaves the published wake benchmark unassessed. The complete
software regression suite takes minutes rather than the full study's hours.
