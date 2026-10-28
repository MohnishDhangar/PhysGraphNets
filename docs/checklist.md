# Development checklist

The installed `.fenicsx-env` is the working CPU Torch/FEniCSx environment.
Clean environment recreation and hosted CI are reproducibility tasks; neither
blocks local numerical development. No reinstall is needed to continue.

## Completed and evidenced

- [x] CPU Poisson and scalar diffusion/reaction FEM, nonzero Dirichlet and mixed boundaries.
- [x] Structured/graded square, annulus, cylinder and airfoil FEM geometry.
- [x] Independent FEniCSx checks for 14 fixtures, including both mixed square orientations.
- [x] NumPy and CPU float64 Torch PINNs on square Dirichlet/mixed cases.
- [x] Exact checkpoint resume and source-compatibility checks.
- [x] Explicit acceptance, controlled multi-seed/collocation study and failure accounting.
- [x] Repeated CPU cost study with common accuracy targets, peak RSS and integration sensitivity.

Evidence: `validation_20261027.md`, `validation_20261028.md`, and
`validation_mixed_cost_20261028.md`.

## Current increment: annulus PINN

- [ ] Domain sampling/membership interface; area-uniform annulus samples.
- [ ] Boundary constraint interface; exact nonzero circle values without analytical interior lifting.
- [ ] Callable coefficient-gradient support for future spatially varying coefficients.
- [ ] Both backends, domain-aware checkpoints and existing CLI integration.
- [ ] True-annulus error quadrature and separately labeled polygon-domain comparison.
- [ ] Derivative parity, manufactured residual checks, exact resume and end-to-end evidence.
- [ ] Full regression suite and original mesh convergence demo.

## Next numerical and ML objectives, in order

- [ ] Broaden irregular-domain cases: annulus mixed conditions, then cylinder/airfoil constraints and samplers.
- [ ] Parameterized PDE-instance dataset schema, reproducible generation and whole-instance splits.
- [ ] Mesh-to-graph builder with edge symmetry, boundary tags and permutation tests.
- [ ] Supervised CPU GNN baseline, exact resume, held-out-instance and resolution evaluation.
- [ ] Broader accuracy/cost studies, repeated inference, and detailed operator profiling.

## Separate execution and product tasks

- [ ] Recreate a clean environment when package downloads are available (previous attempt failed DNS resolution).
- [ ] Execute hosted CI when the source is in a configured remote repository.
- [ ] Execute the same package on actual cloud hardware and record parity.
- [ ] Implement the interactive viewer.
- [ ] Consider GPU work only after a measured bottleneck and explicit hardware/precision/memory reporting.

GNN/GPU dependencies are not introduced by the annulus increment.
