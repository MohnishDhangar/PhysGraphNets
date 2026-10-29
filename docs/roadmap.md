# Research milestones (gated, not calendar deadlines)

The maintained [development checklist](checklist.md) separates local development
from optional environment recreation and hosted execution checks.

October 28 reliability work adds CPU environment snapshots, a CI matrix,
checkpoint source compatibility, configurable evaluation and a multi-seed PINN
study runner. See [implementation and remaining gates](reliability.md).

## Phase 0 — repository/contracts [scaffold complete]
Editable Python package, independent numerical modules, tests, repeatable
configs, cloud bootstrap, research rules. No heavyweight ML dependencies.

## Phase 1 — Poisson FEM + visualization [initial implementation included]
Unit-square P1 FEM, analytical manufactured cases, boundary tests, mesh
convergence and PNG 2D field displays. First review correctness, then freeze.

## Phase 2A — general elliptic CPU reference [IMPLEMENTED]
Scalar variable `k(x)` and reaction `c(x)`, full Dirichlet or mixed
Dirichlet/Neumann boundaries, nonuniform triangular meshes, independent
analytical-reference contracts, mass/reduced operators and experiment manifests.
Manufactured cases and tests cover affine exactness, mixed fluxes, nonzero
forcing/boundaries and convergence. See `docs/phase2.md` for formulation and commands.
Pure Neumann and arbitrary-domain meshing are outside this increment.
Independent FEniCSx verification passed for all 12 fixtures at resolutions 4
and 8 (24 comparisons), plus the integration suite. See `fenicsx.md` and
`validation_20261026.md` for evidence and scope.

### Phase 2A geometry extension — annulus [IMPLEMENTED]
Radial/angular triangular annulus, periodic seam, named inner/outer boundary
edges, analytical logarithmic diffusion and both mixed-boundary orientations.
Validation separates polygonal geometry error from solution error and refines
angular/radial resolution together. Tagged archives use schema v2 while preserving
v1 support. See `docs/annulus.md` and `scripts/run_annulus.py`.

### Phase 2A geometry extension — rectangle with cylinder [IMPLEMENTED]
Conforming O-grid with exact rectangle corners and a polygonal circular hole;
five named boundary groups, quadratic and affine diffusion/reaction references,
all-Dirichlet and both mixed-boundary variants, geometry/solution convergence,
tagged archives and manifests. See `docs/cylinder.md`.

### Phase 2A geometry extension — symmetric airfoil exterior [IMPLEMENTED]
Closed symmetric NACA-style profile with upper/lower surface tags and a circular
far field; rotation, scale, thickness and radial grading; affine/quadratic
manufactured elliptic references and all three boundary modes. Solution errors,
profile-area and far-field-area convergence are reported independently. Tagged
archives and manifests preserve the exact profile convention. See `docs/airfoil.md`.
Independent FEniCSx verification is complete for the tested fixture set.
Actual cloud execution remains open.

## Phase 2B — conventional PINN baseline [NUMPY AND PYTORCH CPU IMPLEMENTED]
Initial implementation now has an independently checked FEM reference. A small
NumPy tanh MLP solves square Dirichlet manufactured cases using analytical spatial
derivatives, complex-step parameter gradients and Adam. Hard boundary lifting,
pickle-free deterministic resume, direct neural L2/H1 quadrature, held-out spatial
residuals, and exact/FEM/PINN artifacts are implemented. The optional PyTorch
backend uses CPU float64 autograd through the full constrained field and native
Adam, with derivative parity checks and deterministic NPZ resume. NumPy remains
the default and needs no new dependencies. See `pinn.md`. Multi-seed studies and
left/right mixed-boundary square cases are implemented; see `reliability.md` and
`mixed_pinn.md`. A Dirichlet annulus PINN now has domain-aware sampling, hard
circle constraints and distinct true-domain/polygon error metrics; see
`annulus_pinn.md`. Other irregular geometries and broader acceptance remain open.

The PyTorch increment passed full regression and four manufactured training
runs; [27 October validation evidence](validation_20261027.md) records scope,
errors and reproducibility checks.

## Phase 3 — graph network [CPU MODEL/TRAINER IMPLEMENTED; SCOPED SQUARE REPAIR QUALIFIED]
Directed mesh graph builder, physical input features, canonical PDE-instance IDs,
whole-instance splits, checksum-validated loading, training-only normalization,
disjoint batching, and FEM-labelled annulus/parameterized-square generation are
implemented. Numerical label audits and P1 prediction metrics are available.
See `graph_data.md`, `gnn_protocol.md` and `validation_pre_gnn.md`.
The supervised encoder–processor–decoder and deterministic epoch resume are implemented.
The first three-seed qualification failed the declared L2/H1/residual gates; see
`validation_gnn_flow_20261029.md`. A global square model with derivative supervision
subsequently passed unchanged gates on 16 fresh instances at n=8/24, all three
seeds; see `gnn_repairs.md`. Next: cost profiling, broader boundary/geometry families
and independently declared extrapolation tests. Train on **multiple PDE instances**,
holding out full instances and mesh resolutions under the declared protocol.
Acceptance: train + resume, clear accuracy baselines, independent graph checks.

## Phase 4 — experimental verification + profiling [INITIAL CPU STUDY IMPLEMENTED]
`scripts/run_pinn_cost.py` runs repeated isolated FEM/Torch trials, records timing
and peak RSS, checks evaluation sensitivity, and selects candidates meeting common
L2/H1 targets. See `cost_study.md`. This is an initial single-instance CPU study;
the broader work below remains open.
PDEs, geometry shifts, mesh refinement, operator learning, measured memory
and latency. Document bottleneck candidates before kernel design.

## Phase 5 — interactive viewer [DEFERRED FINAL PHASE]
Separate web application with 2D scalar fields, triangulated domains,
colormaps, probe values, reference/prediction/error, playback for time PDEs.
Keep Matplotlib exports throughout all phases for independent visual checking.
The [fine-viewer plan](viewer_plan.md) now specifies smooth periodic replay,
dense tracers/trails, dye transport, fixed-geometry flow and later fluid-surrogate
integration, with measurable acceptance gates. User direction defers replay and
video development until the primary numerical and model deliverables qualify. Full-car/moving-wing CFD and free-surface liquids
require additional physics phases; they are not display-only options.

## Phase 6 — custom CUDA kernel [NOT IMPLEMENTED]
Only after Phase 4. Begin with proven bottleneck and explicit forward/backward
operator contract. Benchmark numerical parity and training peak memory against
PyTorch baseline on an available NVIDIA GPU. Report device + environment.

## Reserved module placeholders

`src/physgraph/cuda/` and `viewer/` remain placeholders.
`graph/` and `datasets/` contain the initial graph-data foundations.
`profiling/` contains isolated CPU cost-study workers.
`models/`, `losses/`, and `training/` now contain the initial CPU PINN baseline.

## Fluid/particle direction — manufactured transient flow and stationary obstacles qualified

Boundary-aware fluid physics and passive particle transport are the active
priority; viewer work is deferred to final integration. The periodic incompressible reference, manufactured vortex/shear
checks, tracer convergence and time-indexed GIF export are implemented.
A CPU P2/P1 Stokes solver now passes manufactured square refinement and independent
FEniCSx comparisons; see [formulation and scope](stokes.md). The [mixed-boundary increment](stokes_mixed.md) adds physical traction, absolute
pressure, channel fluxes and independently integrated loads. The [obstacle
increment](obstacle_stokes.md) qualifies initial no-slip cylinder/airfoil fixtures,
including confined-cylinder analytical drag and independent reaction loads.
The [transient increment](navier_stokes.md) adds fully implicit advection,
BE/BDF2, constrained initialization, manufactured refinement, independent
trajectories and closed-wall energy checks. Next are a precisely specified
transient cylinder benchmark, expanded obstacle/domain studies, triangular
particle sampling and transient learned models. Arbitrary-mesh import and
fluid GNN qualification remain open.
See `fluid_particles.md` for equations and scope.
