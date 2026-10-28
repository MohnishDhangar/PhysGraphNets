# PyTorch CPU PINN validation — 27 October 2026

The optional PyTorch backend is implemented and exercised end to end. It solves
the existing square Dirichlet Poisson and scalar diffusion/reaction cases with
CPU float64 autograd and native Adam. This completes the initial backend
increment; mixed neural boundaries, irregular neural domains, multiple seeds,
cloud execution and broader accuracy/cost studies remain open.

## Executed checks

| Check | Result |
|---|---|
| Full suite in `.fenicsx-env` | **187 passed, 1 skipped**, 66.43 s |
| NumPy + PyTorch PINN tests | **46 passed**, 29.60 s |
| Full suite in `.pignn-venv` without Torch/FEniCSx | **151 passed, 13 skipped**, 34.79 s |
| Poisson mesh convergence, n=8,16,32 | Completed with expected L2/H1 rates |
| Four single-seed PyTorch training/evaluation runs | Completed; metrics below |

The full installed-environment run executed all 12 FEniCSx integration cases;
its only skip is the intentional missing-dependency path. The lightweight run
skips 12 FEniCSx cases and the optional PyTorch test module. Neither skip is
counted as a passed numerical check.

The new tests check autograd values, gradients, Laplacians, residuals and
parameter gradients against NumPy on all four cases and three architectures.
They cover nonzero Dirichlet lifting, exact affine residuals, training/error
reduction, short Adam trajectory parity, bit-identical same-environment resume,
incompatible/corrupt checkpoint rejection, runtime/RNG restoration, absence of
analytical-reference leakage into training, and distinct FEM/PINN CLI artifacts.

Two initial full-suite attempts terminated at MPI initialization because this
restricted session prohibits sockets. A diagnostic import identified UCX's
shared-memory transport initialization as the cause. The successful run used
`UCX_TLS=self`, appropriate to this adapter's single-process scope:

```bash
XDG_CACHE_HOME=/tmp/physgraph-cache MPLCONFIGDIR=/tmp/physgraph-mpl \
OPENBLAS_NUM_THREADS=1 UCX_TLS=self \
  "$HOME/.local/bin/micromamba" run -p "$PWD/.fenicsx-env" python -m pytest -q
```

This changes only the runtime transport. No integration tests were disabled.
See [FEniCSx execution notes](fenicsx.md) for the transport documentation.

## Experiment configuration and provenance

- Python **3.12.13**, PyTorch **2.10.0+cpu**, NumPy **2.5.4**, SciPy **1.18.1**.
- CPU float64, one Torch thread, `OPENBLAS_NUM_THREADS=1`, deterministic algorithms.
- Width 8, depth 2, **105 parameters**, 128 fixed collocation points, seed 7.
- Adam learning rate 0.003, betas (0.9, 0.999), epsilon 1e-8.
- Evaluation mesh **16×16**, 289 nodes, 512 positively oriented triangles.
- Continuous neural quadrature order 5; separate 1,024-point residual sample, seed 10007.
- FEM assembly order 4, solved independently of the neural model.

Git metadata is unavailable in this workspace. All four run manifests and the
mesh-convergence manifest record source SHA-256:
`6f25138eadd9fd46fd5ef3400461aeba1bb1ac00160f789cb26928e911dcac7f`.
The date in this document is local (Asia/Kolkata); manifests use UTC.

## Measured neural errors

| Case | Steps | Relative L2 | Relative H1 seminorm | Final normalized training loss | Held-out residual RMS |
|---|---:|---:|---:|---:|---:|
| `sine` | 1000 | 0.004529893 | 0.01111901 | 0.001391182 | 0.4068459 |
| `variable_sine` | 1000 | 0.0007772702 | 0.008701993 | 0.001281044 | 0.9005946 |
| `affine` | 300 | 0.0001868288 | 0.0005226096 | 0.0006766874 | 0.02601126 |
| `affine_diffusion` | 300 | 0.0001967195 | 0.0005748606 | 0.0003351866 | 0.05527670 |

Boundary errors are at most **1.225e-16**. Loss fell from 1.038504, 1.005832,
1.553738 and 0.7874473, respectively. All errors use the analytical references;
the held-out residual measures the same PDE instance away from training points.
The affine FEM solutions remain exact to roundoff: maximum nodal errors were
1.33e-15 and 5.33e-15, while the corresponding learned fields retain nonzero error.

Recorded training wall times were 5.06, 4.01, 1.18 and 1.25 seconds, respectively;
they include model/optimizer setup and checkpoint I/O. FEM assembly/solve times
were 0.0227–0.0340 seconds. These are single runs with different approximation
spaces and no matched-accuracy timing study; they do not establish a performance
advantage. Continuous neural errors are distinct from interpolation errors of
the plotted nodal fields.

## Mesh convergence

| n | Relative L2 | Relative H1 seminorm |
|---|---:|---:|
| 8 | 0.04221224 | 0.19437753 |
| 16 | 0.01075142 | 0.09792576 |
| 32 | 0.002700655 | 0.04905618 |

The final refinement rates are approximately **1.993 in L2** and **0.997 in H1**.
This demo uses the original Phase 1 assembly quadrature; the PINN experiment's
FEM comparison uses the separate higher-order Phase 2 assembly configuration.

## Saved evidence and reproduction

[Aggregate manifest](../outputs/torch_pinn_validation_20261027/validation_manifest.json),
[mesh convergence manifest](../outputs/torch_pinn_validation_20261027/convergence/convergence_manifest.json),
and per-case manifests:
[sine](../outputs/torch_pinn_validation_20261027/sine/manifest.json),
[variable sine](../outputs/torch_pinn_validation_20261027/variable_sine/manifest.json),
[affine](../outputs/torch_pinn_validation_20261027/affine/manifest.json),
[affine diffusion](../outputs/torch_pinn_validation_20261027/affine_diffusion/manifest.json).
Each case has NPZ checkpoint/reference/comparison archives, training CSV, metrics
JSON and comparison/training plots. Generated outputs remain ignored by version control.

```bash
"$HOME/.local/bin/micromamba" run -p "$PWD/.fenicsx-env" \
  python scripts/run_pinn.py --config configs/experiments/pinn_torch_cpu.json
```

Change `--case` to reproduce other cases; use `--steps 300` for the two affine
runs. CLI defaults create unique output directories. The full mathematical
formulation, resume commands and remaining scope are in [pinn.md](pinn.md).
