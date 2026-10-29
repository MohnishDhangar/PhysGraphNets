# GNN repair evidence — 29 October 2026

The repaired CPU model passed the original acceptance thresholds for all three
seeds on 16 newly declared test instances at n=8 and n=24. This qualifies only
the tested steady scalar parameterized square family and its affine Dirichlet /
right-Neumann boundary modes. It does not qualify fluid prediction, arbitrary
geometry, parameter extrapolation or production accuracy.

## Problems and implemented responses

The original three-hop architecture could not transmit distant boundary data to
much of a refined mesh. Its nodal MSE objective and MSE checkpoint selection also
did not directly optimize the gradient error required for acceptance. A small
last batch and fixed learning rate were optimization concerns, not proven sole
causes. Independent P1 metric recomputation agreed with evaluation, so changing
the evaluator or relaxing the gates was not justified.

Implemented:

- A scoped global anchor graph, smooth coordinate decoder and boundary-data-only
  affine lifting, giving all nodes access to distant physical/boundary inputs.
- Differentiable, exact triangle-integrated relative P1 L2/H1 supervision and
  checkpoint selection using that declared objective.
- Deterministic learning-rate decay, batch size 7 for 49 training instances, and
  checkpoint schema 2 with exact resumed-state equivalence tests.
- Matched validation ablations and a fresh final holdout, excluding all six
  previously inspected test instances. Acceptance thresholds remain unchanged.

The global architecture combines communication and smooth boundary lifting.
Its ablation does not distinguish the contribution of each component. No PDE
residual training loss, GPU dependency or analytical interior solution was added.
See [implementation/scope](gnn.md) and [qualification configuration](../configs/experiments/gnn_repair_qualification.json).

## Validation-only evidence

All rows below use the same nine validation instances at n=8. Matched ablations
use seed 0, width 32, depth 3, 300 epochs, batch 7, Adam .001, with factor-.5 decay
every 100 epochs. The checkpoint is selected using each run's declared objective.

| Model/objective | Selected epoch | Mean relative L2 | Mean relative H1 | Mean equation residual |
| --- | ---: | ---: | ---: | ---: |
| Local / MSE | 189 | 0.068245 | 0.657794 | 0.211577 |
| Local / L2+H1 | 284 | 0.072759 | 0.405951 | 0.080111 |
| Global square / MSE | 289 | 0.032918 | 0.279307 | 0.059956 |
| Global square / L2+H1 | 297 | 0.032035 | 0.232630 | 0.044037 |
| Global square / L2+H1, extended to 800 epochs | 793 | 0.027283 | 0.186421 | 0.036559 |

The extended run has a larger optimization budget and is not a matched-budget
architecture comparison. It met the original thresholds on validation, after
which the final 800-epoch configuration was frozen. A separate one-instance
diagnostic reached L2=0.007388 and H1=0.067537 after 300 steps. No test predictions
were used for these decisions. Ablation manifests assess baseline improvement;
their `passed` flag with empty numeric criteria is not final qualification.

Artifacts, preserved under ignored `outputs/gnn_repairs_20261029/`:

- `one_instance_diagnostic.json`
- `ablations/manifest.json` and each run's checkpoint/predictions
- `validation_extension_manifest.json`
- `global_h1_validation_extension/checkpoint.pt`

## Fresh final qualification

`configs/experiments/gnn_repair_holdout_protocol.json` declares the 16 new test
IDs before label generation or test prediction. They are the first eight hash-test
parameter proposals per boundary mode under the declared RNG seed, within the
original parameter ranges. The 49 training and nine validation identities are
unchanged, as are the n=8 training normalization statistics (checked exactly).
Each new test instance is evaluated at n=8 and n=24; n=24 was not used for selection.

All seeds 0/1/2 trained for the frozen 800 epochs before test evaluation began.
Every seed is reported. Gates for each resolution remain mean L2<=.05, max L2<=.1,
mean H1<=.2, max H1<=.4, mean assembled equation residual<=.1 and max Dirichlet
error<=1e-12, plus improvement over the boundary-affine baseline in mean L2/H1.

| Seed | n | Mean L2 | Max L2 | Mean H1 | Max H1 | Mean equation residual | Result |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 8 | .019641 | .038780 | .182424 | .375516 | .033101 | Pass |
| 0 | 24 | .020514 | .040433 | .183648 | .361832 | .008952 | Pass |
| 1 | 8 | .015892 | .034447 | .149553 | .350934 | .027968 | Pass |
| 1 | 24 | .017220 | .036156 | .156055 | .346580 | .007940 | Pass |
| 2 | 8 | .017235 | .030726 | .166727 | .371630 | .030305 | Pass |
| 2 | 24 | .017953 | .030954 | .169810 | .359837 | .008349 | Pass |

Maximum Dirichlet error is zero in every group. FEM values remain the reference.
The lower algebraic residual on n=24 is not by itself evidence of a continuum
convergence rate; its normalization depends on the assembled discrete system.
The original failed study remains historical evidence on a different test set,
so old/new test means are not a controlled paired effect estimate.

Final artifacts: `fresh_dataset/dataset_manifest.json`, `fresh_normalization.json`
and `fresh_qualification/manifest.json`, with per-seed schema-2 checkpoints,
FEM/learned/baseline predictions and per-instance metrics. Manifests record Python,
dependencies, CPU float64, source identity, dataset checksums and wall time.

## Executed verification

- Installed Python 3.12.13 / CPU Torch 2.10.0 / FEniCSx environment: full
  `python -m pytest -q`: **281 passed, 1 expected skip** in 83.37 s.
- Base Python 3.14.7 environment without optional Torch/FEniCSx: full suite:
  **226 passed, 26 expected skips** in 38.87 s.
- Original mesh convergence demo, n=8/16/32: relative L2
  .042212/.010751/.002701 and relative H1 .194378/.097926/.049056, consistent
  with the expected approximately second-/first-order behavior.
- Offline wheel build: succeeded without downloading dependencies.
- New tests cover exact loss/evaluator agreement, autograd gradient checking,
  zero target error, constant-offset gradients, permutation equivariance,
  independent batched graphs, exact nonzero prescribed values, remote-boundary
  sensitivity and exact model/Adam/history resume through learning-rate changes.

## Remaining work

Repeated inference/training cost and peak-memory profiling, larger held-out
populations, extrapolation, non-affine boundary data and other geometries remain
open. This model's nine fixed anchors and unit-square boundary envelope are not
a general mesh architecture. Future changes must again use training/validation
for tuning; these 16 test instances have now been examined.

The next product work is [fine viewer playback](viewer_plan.md), alongside a
boundary-aware fluid reference. A transient fluid GNN needs a separate formulation,
dataset and multi-step physical qualification.
