# Phase 1 verification and acceptance gates

PDE: `-Δu=f` on `[0,1]^2`, prescribed Dirichlet boundary values.

## Tests

1. Mesh: `(nx+1)*(ny+1)` nodes, `2*nx*ny` triangles, positive signed areas,
   combined area 1.
2. FEM: symmetric matrix; positive-definite reduced interior operator.
3. Boundaries: affine exact field `u=1+2x-3y` passes on *all* nodes to roundoff,
   proving nonzero Dirichlet values are imposed correctly.
4. Residual: interior algebraic `||Ku-F||/||F-K_IB g_B|| < 1e-10` on small tests.
5. Mesh refinement: smooth sine case L2 and H1 errors decrease with refinement;
   P1 expected rates approach 2 in L2 and 1 in H1 seminorm under appropriate
   regularity and integration accuracy.
6. Exports: compressed fields round-trip; field comparison PNG and metrics
   JSON exist on disk.
7. Cloud parity: source/tests/smoke demo run on Colab or Kaggle CPU.

Higher-order element quadrature gives approximate continuum errors. Do not interpret
near-zero FEM *algebraic* residual as a zero *PDE* approximation error.

## Phase 2A verification

See `phase2.md` for the formulation, manufactured families and acceptance tests.
Run the full suite and the original mesh demo after numerical changes, then
`python scripts/run_phase2.py --config configs/experiments/elliptic_local.json`
and a mixed-boundary refinement run. Phase 2A manifests preserve configurations,
errors and provenance, and convergence CSVs report observed rates at each step.
Independent FEniCSx comparison and actual cloud execution remain separate gates;
local test success does not establish either one.

## Required artifacts before Phase 2

- Passing tests log.
- `comparison.png` for the sine and affine cases.
- `convergence.csv`, `convergence.png` for at least 8,16,32 grids.
- Recorded Python dependency environment and code commit.

## Independent comparison and neural checks

`tests/test_fenicsx_comparison.py` contains pure mapping/fixture checks and 12
external integration cases. The integration cases skip explicitly if FEniCSx
is missing; skipping does not close the independent solver gate. The standalone
comparison CLI instead exits nonzero and saves a blocked manifest.

PINN tests independently check spatial chain rules, complex-step parameter
gradients, the variable-diffusion flux term, nonzero hard boundaries, measurable
training/error reduction, and bit-identical checkpoint resume. Neural evaluation
uses its continuous field/gradient; the saved FEM reference is never overwritten.

`tests/test_torch_pinn.py` checks float64 automatic differentiation against the
NumPy field, residual and complex-step parameter gradients on all four cases
and three architectures. It also checks Adam trajectory parity, exact resume,
checkpoint compatibility, zero affine residuals, boundary lifting, runtime/RNG
isolation, and separate FEM/PINN CLI artifacts. This module skips explicitly
when optional PyTorch is absent; base tests cover missing-Torch behavior.

## Completed independent fixture verification — 26 October 2026

The user ran the full suite in `.fenicsx-env`: 161 passed and one intentional
missing-dependency-path skip. All 12 FEniCSx integration cases executed, and
the comparison runner passed all 24 runs at resolutions 4 and 8. The original
Poisson refinement demo passed as well. The saved manifests and numerical
summary are linked in [the validation record](validation_20261026.md).
This closes the independent FEM gate for these fixtures, while actual cloud
execution and verification of future extensions remain separate requirements.
