# Research milestones (gated, not calendar deadlines)

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
`mixed_pinn.md`. Irregular neural domains and broader acceptance remain open.

The PyTorch increment passed full regression and four manufactured training
runs; [27 October validation evidence](validation_20261027.md) records scope,
errors and reproducibility checks.

## Phase 3 — graph network [NOT IMPLEMENTED]
GraphData builder from Mesh2D; supervised encoder–processor–decoder GNN;
physics-based residual and energy losses. Train on **multiple PDE instances**,
not one forcing field; hold out full instances and mesh resolutions.
Acceptance: train + resume, clear accuracy baselines, independent graph checks.

## Phase 4 — experimental verification + profiling [INITIAL CPU STUDY IMPLEMENTED]
`scripts/run_pinn_cost.py` runs repeated isolated FEM/Torch trials, records timing
and peak RSS, checks evaluation sensitivity, and selects candidates meeting common
L2/H1 targets. See `cost_study.md`. This is an initial single-instance CPU study;
the broader work below remains open.
PDEs, geometry shifts, mesh refinement, operator learning, measured memory
and latency. Document bottleneck candidates before kernel design.

## Phase 5 — interactive viewer [NOT IMPLEMENTED]
Separate web application with 2D scalar fields, triangulated domains,
colormaps, probe values, reference/prediction/error, playback for time PDEs.
Keep Matplotlib exports throughout all phases for independent visual checking.

## Phase 6 — custom CUDA kernel [NOT IMPLEMENTED]
Only after Phase 4. Begin with proven bottleneck and explicit forward/backward
operator contract. Benchmark numerical parity and training peak memory against
PyTorch baseline on an available NVIDIA GPU. Report device + environment.

## Reserved module placeholders

`src/physgraph/graph/`, `cuda/`, and `viewer/` remain placeholders.
`profiling/` contains isolated CPU cost-study workers.
`models/`, `losses/`, and `training/` now contain the initial CPU PINN baseline.
