# Development checklist

The installed `.fenicsx-env` is the working CPU Torch/FEniCSx environment.
Clean environment recreation and hosted CI are reproducibility tasks; neither
blocks local numerical development. No reinstall is needed to continue.
Active priority: boundary-aware fluid physics, reference validation, datasets and
model qualification. Viewer and final-video work are deferred until those primary
deliverables qualify. Latest numerical milestone: [manufactured transient
Navier–Stokes](navier_stokes.md), extending the [stationary obstacle reference](obstacle_stokes.md).
See [executed transient validation](validation_navier_stokes_20261030.md).

Earlier evidence: [GNN repair and fresh qualification](gnn_repairs.md), following
the [initial GNN/fluid validation](validation_gnn_flow_20261029.md).

## Completed and evidenced

- [x] CPU Poisson and scalar diffusion/reaction FEM, nonzero Dirichlet and mixed boundaries.
- [x] Structured/graded square, annulus, cylinder and airfoil FEM geometry.
- [x] Independent FEniCSx checks for 16 fixtures, including mixed square orientations and the two-mode square family.
- [x] NumPy and CPU float64 Torch PINNs on square Dirichlet/mixed cases.
- [x] Exact checkpoint resume and source-compatibility checks.
- [x] Explicit acceptance, controlled multi-seed/collocation study and failure accounting.
- [x] Repeated CPU cost study with common accuracy targets, peak RSS and integration sensitivity.

Evidence: `validation_20261027.md`, `validation_20261028.md`, and
`validation_mixed_cost_20261028.md`.

## Completed extension: annulus Dirichlet PINN

- [x] Domain sampling/membership interface; area-uniform annulus samples.
- [x] Boundary constraint interface; exact nonzero circle values without analytical interior lifting.
- [x] Callable coefficient-gradient support for future spatially varying coefficients.
- [x] Both backends, domain-aware checkpoints and existing CLI integration.
- [x] True-annulus error quadrature and separately labeled polygon-domain comparison.
- [x] Derivative parity, manufactured residual checks, exact resume and end-to-end evidence.
- [x] Full regression suite and original mesh convergence demo.

The initial 3,000-step run missed the declared residual threshold; an extension
to 6,000 total steps passed it unchanged. Both results are retained. A separate
8,192-point assessment and quadrature-sensitivity check also completed.

## Completed pre-GNN data and numerical gates

- [x] Initial parameterized annulus dataset schema, reproducible FEM-label generation and whole-instance splits.
- [x] Mesh-to-graph builder with edge symmetry, boundary tags and permutation tests.
- [x] Checksum/schema validation, complete instance/refinement coverage and split checks.
- [x] Training-only, resolution-restricted normalization and deterministic disjoint batching.
- [x] Richer two-mode square diffusion/reaction family with Dirichlet and mixed boundaries.
- [x] FEM label reassembly, quadrature sensitivity, convergence and independent FEniCSx checks.
- [x] P1 prediction metrics and a declared held-out-instance/resolution protocol.
- [x] Expanded 64-instance dataset; reserve n=16 before training at n=8.
- [x] Offline package wheel and import/preprocessing checks in both local environments.

## Next deliverable: supervised CPU GNN, in dependency order

Reviewed against the code on 29 October 2026. A supervised CPU GNN model and epoch-based trainer are now implemented; see
[implementation and scope](gnn.md). The graph-data audit is complete. The first study failed; the scoped global-model repair passed unchanged gates on a fresh three-seed holdout. Broader learned-model acceptance remains open. Follow [the declared
protocol](gnn_protocol.md); the existing CPU PyTorch environment is sufficient.

- [x] Implement a CPU float64 node/edge encoder, message-passing processor and scalar decoder.
- [x] Enforce prescribed Dirichlet values while predicting interior and Neumann-node values.
- [x] Verify model permutation equivariance, disjoint-batch isolation, finite gradients and boundary enforcement.
- [x] Implement supervised optimization with a declared mesh-aware loss and equal physical-instance weighting.
- [x] Implement checkpoints covering model, optimizer, RNG, epoch/batch progress, source, dataset and normalization provenance; prove uninterrupted/resumed equivalence.
- [x] Add experiment config and CLI; enforce n=8 training/validation selection and reserve n=16 for subsequent evaluation.
- [x] Declare seeds, stopping rule, model acceptance thresholds and simple boundary-data baselines before inspecting test predictions.
- [x] Execute the first controlled training study, preserving all seeds and failures.
- [x] Evaluate held-out physical instances at n=8 and n=16 separately using FEM-relative L2/H1, boundary error and equation residual; report per-instance spread and worst cases.

The original dataset has 49 training, 9 validation and 6 test instances. The repair
retains training/validation identities and replaces the old test set with 16 fresh
instances evaluated at n=8/24. It is an
initial benchmark; exact-identity separation does not establish parameter or
geometry extrapolation. Label-quality thresholds are not GNN acceptance targets.

## GNN qualification follow-up

- [x] Implement and compare derivative-aware supervision and global communication on validation only; declare fresh final instances and n=24 refinement.
- [x] Pass unchanged GNN accuracy/residual criteria for the scoped square family: all three seeds passed fresh n=8/24 tests. Broader geometry/physics qualification remains open.
- [ ] Measure training cost and repeated inference separately; report CPU precision, timing, peak memory and FEM baselines.
- [ ] Profile operators and identify measured bottlenecks before hardware-specific optimization.
- [ ] Expand instance counts, forcing/coefficient families and declared extrapolation tests; broaden graph geometry evaluation beyond the square family.

## Fluid flow and particle motion — active next product direction

Fluid physics and particle transport are prioritized. Viewer work, including
heat/diffusion presentation, is on hold as a final integration step.

- [x] Formulate periodic incompressible flow and exact vortex/shear cases.
- [x] Implement dealiased Fourier vorticity solver with RK4 and stability checks.
- [x] Implement passive tracer integration, periodic interpolation and exact trajectory checks.
- [x] Verify divergence, nonlinear RHS, energy decay, temporal and particle spatial convergence.
- [x] Export time-indexed numerical fields/particles and a verified GIF playback artifact.
- [x] Document a [fine-viewer implementation plan](viewer_plan.md), including the limits of accessible video references.
- [x] Inspect six local reference videos and review the three linked repositories; record [source findings and reuse decisions](viewer_reference_review.md).
- [ ] Import arbitrary supported meshes with complete boundary tags and quality checks.
- [x] Implement CPU P2/P1 Stokes with full velocity Dirichlet conditions, pressure gauge, manufactured cases and independent FEniCSx comparisons.
- [x] Add exhaustive velocity/traction partitions, physical symmetric-gradient stress and traction-determined pressure; verify known-traction channel boundaries and independent flux/load integration.
- [x] Check pressure shifts, all four boundary normals, edge/node permutations, hole normals and zero-traction manufactured extension flow.
- [x] Qualify initial finite-domain no-slip cylinder/airfoil Stokes fixtures with independent fields, fluxes, surface/reaction loads and refinement checks.
- [x] Add generated-mesh topology/quality audits and boundary-edge resolution reporting; verify confined-cylinder analytical fields/drag at three outer radii.
- [ ] Extend obstacle parameter coverage, channel outlet-length and airfoil outer-radius sensitivity; the initial fixtures do not establish arbitrary-geometry accuracy.
- [x] Implement fully implicit manufactured transient Navier-Stokes; verify BE/BDF2 time order and P2/P1 spatial convergence before obstacle wake studies.
- [x] Independently compare initial projections and complete manufactured trajectories, including physical traction, absolute pressure and inertial force balance.
- [x] Verify closed-wall decay and discrete BE/BDF2 energy identities, retaining advective power from weak incompressibility.
- [ ] Extend to transient flow around obstacles with validated wall/inflow/outflow treatment.
- [ ] Add triangle-based particle sampling and explicit boundary interaction policies.
- [ ] Build transient vector-field datasets, models and long-rollout evaluation.
- [ ] Add transient simulation restart/checkpoints and streaming output for longer runs.
- [ ] Formulate and validate transported dye/temperature as distinct physical fields.
- [ ] Final phase (deferred): smooth browser playback, seam handling, dense display tracers/trails, emitters and deterministic video export.
- [ ] Final phase (deferred): integrate fluid/particle playback into the interactive viewer; add heat/diffusion afterwards.

See [formulation and present scope](fluid_particles.md). Periodic-box validation
is not evidence for arbitrary shapes, fluid GNN accuracy or moving solids.

## Separate numerical extensions

- [ ] Extend annulus PINNs to mixed boundary conditions.
- [ ] Implement cylinder/airfoil PINN constraints and domain samplers, with numerical validation and controlled studies.

These PINN extensions do not block the initial supervised square-domain GNN.

## Separate execution and product tasks

- [ ] Recreate a clean environment when package downloads are available (previous attempt failed DNS resolution).
- [ ] Execute hosted CI when the source is in a configured remote repository.
- [ ] Execute the same package on actual cloud hardware and record parity.
- [ ] Implement the interactive viewer.
- [ ] Consider GPU work only after a measured bottleneck and explicit hardware/precision/memory reporting.

No graph-learning framework or GPU dependency has been added. Custom GPU kernels
remain conditional on a verified model, profiling evidence and a phase decision.
