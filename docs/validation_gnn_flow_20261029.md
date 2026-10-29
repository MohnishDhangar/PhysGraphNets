# GNN qualification and first fluid/particle milestone — 29 October 2026

GNN scientific qualification **failed** its predeclared acceptance criteria.
The independent periodic fluid reference and tracer convergence checks passed.
These are distinct results: scalar steady elliptic learning does not qualify a
Navier-Stokes surrogate. The installed full suite passed **277 tests, 1 expected
missing-dependency-path skip** in 86.01 seconds; the original 8/16/32 Poisson mesh
convergence demo retained its established errors.

## Controlled GNN qualification

Seeds 0/1/2 each trained for the fixed 120 epochs on the 49 n=8 training instances.
All three runs finished before test evaluation. Checkpoints were selected using
only the 9 n=8 validation instances, at epochs 106/95/113 respectively. All seeds
were evaluated on the same six held-out physical instances, at n=8 and n=16.
The n=16 data were excluded from fitting, optimization and model selection.

| Seed | Mean L2, n=8 | Mean H1, n=8 | Mean residual, n=8 | Mean L2, n=16 | Mean H1, n=16 | Mean residual, n=16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.06624 | 0.48157 | 0.18848 | 0.08855 | 0.71378 | 0.18604 |
| 1 | 0.06534 | 0.44732 | 0.16541 | 0.08791 | 0.68018 | 0.16740 |
| 2 | 0.06342 | 0.46388 | 0.17029 | 0.09082 | 0.67494 | 0.15700 |
| Required | <=0.05 | <=0.20 | <=0.10 | <=0.05 | <=0.20 | <=0.10 |

L2/H1 are relative P1 field errors against independent FEM labels. Every seed
also missed the maximum H1 criterion; n=16 missed maximum L2. Worst n=16 nodal
field L2 over all seeds/instances was 0.20365, and worst H1 was 1.31861. Prescribed
Dirichlet error was exactly zero throughout. These gradient errors are material:
visually plausible scalar fields are insufficient evidence of a PDE solution.

The boundary-affine baseline fits only physical Dirichlet data. Its mean L2/H1
were 0.24911/0.98334 at n=8 and 0.25466/0.98400 at n=16. All learned models improve
those averages, but baseline improvement alone does not satisfy acceptance.
The report includes per-instance scores, median, maximum and population standard
deviation; six test instances do not support broad generalization claims.

Artifacts: `outputs/gnn_qualification_20261029/manifest.json`, all three seed
checkpoints/history manifests, and per-instance numerical predictions, FEM labels
and baseline values in `evaluation_*/`. The failed study exits nonzero. No test
seed was selected or discarded, thresholds were not relaxed, and no post-test
extension was performed. Study elapsed time was 84.14 seconds in this execution;
this includes validation/I/O and is not an isolated performance benchmark.

Next model work should use training/validation data: investigate derivative-aware
supervision or residual losses, and test whether global/multiscale communication
improves resolution transfer. The three-layer graph's physical receptive field
shrinks with mesh refinement; this is an architectural concern, not a proven
causal explanation without ablation. Subsequent final evidence must explicitly
identify this test set as reused or use a newly declared final holdout.

## Fluid reference and particle playback

A zero-mean incompressible Fourier vorticity solver now evolves periodic 2D flow
with dealiased nonlinear advection, viscosity and explicit RK4. Passive tracers
use the numerical flow at coupled RK stages with periodic bilinear interpolation.
See [formulation and limits](fluid_particles.md).

The n=32 Taylor-Green run used nu=.05, dt=.01, 200 steps and 100 seeded particles.
Maximum relative velocity L2 error over stored frames was 2.787e-15; maximum
numerical divergence was 4.477e-15. This low-mode exact fixture is deliberately
simple; it does not establish arbitrary-flow, wall or obstacle accuracy. A separate
analytic product-rule test covers nonzero nonlinear vorticity advection.

| Convergence check | Measured result |
| --- | --- |
| Taylor-Green RK4 dt=.05/.025/.0125, t=1 | Relative velocity errors 5.423e-10 / 3.361e-11 / 2.092e-12 |
| Temporal orders | 4.012, 4.006 |
| Exact shear-tracer RMS error, n=16/32/64 | 0.003542 / 0.0009578 / 0.0002530 |
| Particle spatial orders | 1.887, 1.921 |
| Separate n=32 shear trajectory, t=1 | Maximum particle position error 0.004371 |

Both wrapped and unwrapped particle coordinates are retained. The exported
Taylor-Green GIF contains 41 frames at 600x600, with fixed vorticity color limits
and physical timestamps. The final rendered frame was visually inspected. This
is a trajectory/animation export milestone; the interactive viewer is unbuilt.

Artifacts under `outputs/fluid_particles_20261029/`:

- `taylor_green/trajectory.npz`, `particles.gif`, `preview.png`, `manifest.json`.
- `shear/trajectory.npz` and `manifest.json`.
- `convergence_manifest.json`: independent exact shear trajectories and temporal
  reference checks, seeds/parameters, observed orders and declared convergence gates.
- `fem_regression/convergence_manifest.json`: unchanged FEM regression evidence.

Scientific manifests record Python, source fingerprint, code revision, geometry,
time/grid settings and errors. Generated outputs remain ignored. No GNN-generated
fluid labels, pressure solution, obstacle treatment, heat solver, interactive
viewer, hosted CI, cloud execution or GPU result is claimed by this increment.

The lightweight Python 3.14 environment passed **226 tests, 25 optional
dependency skips**. The offline package wheel was built successfully in the
installed environment; a separate import and numerical flow smoke check ran
from the wheel using the lightweight environment. See
`outputs/fluid_particles_20261029/build_manifest.json`.
