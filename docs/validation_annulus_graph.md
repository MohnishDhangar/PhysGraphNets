# Annulus PINN and graph-data validation — October 2026

## Executed tests

- Final installed-environment suite: **224 passed, 1 expected skip**, 80.63 seconds.
- All 14 FEniCSx integration fixtures executed in that suite.
- Before graph additions, the lightweight suite passed **172 tests, 22 expected skips**.
- The seven new graph/data tests then passed in the lightweight environment.
- Focused PINN suite: **64 passed**.
- Original mesh convergence demo passed after the annulus and graph changes.
- 32 graph archives were generated and inspected for identity, split consistency and finite FEM labels.

## Annulus experiment

CPU float64 Torch 2.10.0+cpu; Python 3.12.13; width 16/depth 2 (337 parameters);
256 area-uniform collocation points; seed 7; Adam learning rate 0.003.
FEM mesh: 8 radial cells, 64 angular sectors, 576 nodes, 1,024 triangles.
Neural errors use true-annulus quadrature; separate polygon-domain neural errors
are stored for comparison to FEM. Inner/outer circle values are 1 and 2.

| Total steps | Relative neural L2 | Relative neural H1 | Held-out residual RMS | Acceptance |
|---|---:|---:|---:|---|
| 3000 | 0.000756944246 | 0.00661817161 | 0.256454247 | failed |
| 6000 | 0.000495108899 | 0.00450733269 | 0.187459991 | passed |

The targets were L2<=0.01, H1<=0.05, boundary error<=1e-12 and residual RMS<=0.2.
The 3,000-step run failed the residual criterion. Its checkpoint was extended
adaptively to 6,000 total steps with unchanged thresholds. Both manifests are
preserved. This budget selection used one seed and is not an unbiased multi-seed
performance assessment. Boundary error was 2.22e-16 in both runs.

The extended field was assessed separately at seed 30007 on 8,192 points:
held-out residual RMS **0.1839489053**. L2/H1 errors were stable across true-domain
quadrature orders 3,5,7 and polygon meshes with 8 or 16 radial cells. The radial
and angular quadrature counts scale with order; no analytical interior values
are supplied to the network or loss.

## Graph dataset

16 distinct annulus physical instances, each at two resolutions, produce 32
FEM-labelled graph samples. The split contains **12 train, 1 validation, 3 test
instances** (24/2/6 mesh samples). Every mesh of an instance shares its split.
Raw inputs carry geometry, coefficients and prescribed boundary data; analytical
interior fields and neural predictions are absent. Labels come from CPU FEM.
Maximum recorded algebraic FEM residual over all samples: 6.89761e-15.

This is a contract-validation dataset with a simple radial family. One validation
instance is insufficient for strong model-selection claims. No GNN has been
implemented or trained in this increment. Broader families, a validated dataset
loader, training-only normalization and batching precede that step.

## Provenance and artifacts

Annulus training source snapshot: `9848e27f804694f158c6c959a796157b568b5442f291e14c0af9865e49d7bd2b`.
Final graph/data and assessment source snapshot: `5711765ac77f5402c0ae99c5e59b3ad1dac9575bdf326ebb24fa03d41f79183d`.
Graph/data additions changed the whole-project fingerprint after annulus training;
the training-source fingerprint still matches and checkpoint compatibility was checked.
Git metadata is unavailable; source hashes and package versions are retained.

[Initial annulus run](../outputs/annulus_pinn_20261028/seed7/manifest.json),
[extended annulus run](../outputs/annulus_pinn_20261028/seed7_extended/manifest.json),
[separate assessment](../outputs/annulus_pinn_20261028/assessment_manifest.json),
[graph dataset](../outputs/annulus_graph_20261028/dataset_manifest.json),
[final convergence](../outputs/annulus_graph_20261028/convergence/convergence_manifest.json).

Generated artifacts remain ignored by version control. Working CPU dependencies
are installed; environment recreation and hosted/cloud execution are separate
optional verification tasks, not blockers for these local developments.

[Checklist](checklist.md), [annulus formulation/commands](annulus_pinn.md),
and [graph-data schema/commands](graph_data.md).
