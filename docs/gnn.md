# Supervised CPU GNN implementation

`models/gnn.py` implements a float64 CPU encoder, residual message-passing
processor with mean aggregation, and scalar decoder. Prescribed Dirichlet values
are enforced exactly. Model arguments contain physical features and boundary data,
never FEM targets. No additional runtime dependency was added.

This remains the `local` architecture and default baseline. The repair introduces
`global_square`, described below; it is a scoped alternative, not an
arbitrary-geometry replacement.

`training/gnn.py` uses Adam and lumped-area-weighted squared FEM error, averaged
equally over instances. The initial trainer requires one training resolution per
instance. Validation uses only that resolution; test instances are not evaluated.
The best validation model and latest model are both retained. Epoch checkpoints
include optimizer, Torch RNG, history, dataset/normalization checksums, source
fingerprint and configuration. Shuffling is derived from seed and epoch.
Resume is supported at complete epoch boundaries, not within a batch. Checkpoints
load through Torch's restricted `weights_only=True` interface; use trusted local
artifacts. Training completion does not imply scientific acceptance.

```bash
.fenicsx-env/bin/python scripts/run_gnn.py \
  --dataset outputs/pre_gnn_20261028/square_dataset_extended/dataset_manifest.json \
  --normalization outputs/pre_gnn_20261028/square_audit_extended/normalization.json \
  --epochs 10 --output outputs/my_gnn
```

Resume into a fresh directory with `--resume outputs/my_gnn/checkpoint.pt` and
an increased total epoch count. The package API additionally exposes width,
depth, batch size, seed and learning rate. CPU threading is fixed to one during
training for reproducibility. See `gnn_protocol.md` for experiment restrictions.

## Gap to arbitrary geometry and motion

The current reference solves steady scalar elliptic PDEs on supported triangle
meshes. Geometry alone does not specify a PDE: coefficients, sources, boundary
conditions, and for transient problems initial conditions are required.

Remaining end-to-end work:

1. Geometry/mesh ingestion: contour or mesh imports, holes, boundary tagging,
   triangulation quality checks and supported-geometry validation.
2. General graph data import: current loader reconstructs canonical square and
   annulus recipes; extend it to externally supplied meshes and physical fields.
3. GNN scientific qualification: declared accuracy targets, baselines, multiple
   seeds, held-out mesh/parameter/geometry studies and inference cost measurements.
4. Time-dependent PDE formulation and manufactured tests, stable time integration,
   reference trajectories and rollout validation. A steady scalar field cannot
   by itself produce a physically justified motion sequence.
5. Define what moves: scalar diffusion animations, advected particles, fluid flow,
   or deforming solids require different equations and state representations.
6. Frame/trajectory artifacts with timestamps, vector/scalar fields and provenance;
   consistent color scales, rendering and video export.
7. Interactive viewer with mesh/boundary inputs, solver controls, field inspection,
   playback and experiment comparison; then integrated end-to-end tests.

An initial static elliptic mesh-to-field tool is substantially closer than the
full arbitrary-shape motion product. A percentage for that broader goal would be
misleading until supported geometry, PDE families and motion semantics are fixed.

## Original accuracy qualification

`configs/experiments/gnn_qualification.json` fixes seeds 0/1/2, 120 epochs,
width 32, three message-passing blocks, batch size 8 and Adam learning rate .001.
All seeds train before test evaluation. Best checkpoints are selected within
seed on n=8 validation MSE; no seed is selected by test accuracy. A validated,
immutable sample cache includes only training/validation instances at the fit
resolution. It avoids repeated mesh reconstruction each epoch; it does not cache
reserved-resolution or test labels for training.

The runner reports each test instance and separate per-resolution mean, median,
maximum and standard deviation (population, not confidence intervals). Each
physical instance has equal weight. The fixed gates for BOTH n=8 and n=16 are:
mean relative FEM L2 <=.05, maximum L2 <=.1, mean H1 <=.2, maximum H1 <=.4,
mean assembled equation residual <=.1, and maximum Dirichlet error <=1e-12.
Mean L2 and H1 must also improve a least-squares affine extension fitted only to
prescribed boundary values. All seeds must pass; failures remain in the manifest.
These are development criteria on six test instances, not production guarantees.

```bash
.fenicsx-env/bin/python scripts/run_gnn_study.py \
  --config configs/experiments/gnn_qualification.json \
  --output outputs/my_gnn_qualification
```

A failed qualification exits nonzero and preserves checkpoints, all predictions,
FEM labels, baseline values and manifests. Do not weaken thresholds or extend a
run after examining test results to report the same test set as fresh evidence.
Design subsequent changes on training/validation and identify reused test data
explicitly or generate a newly declared final test set.

## Validation-driven repair

`models/global_gnn.py` uses area-weighted messages from physical nodes to nine
fixed coarse anchors, followed by a global context and smooth coordinate decoder.
Every predicted node can depend on remote boundary data. An affine lifting is
fitted only to prescribed values; a smooth polynomial envelope enforces the
Dirichlet boundary without an abrupt correction on the first free-node row.
No manufactured solution, FEM target or case parameter enters model inference.

This model is deliberately restricted to the unit square, affine Dirichlet data
on left/top/bottom, and either Dirichlet or Neumann data on the right. Unsupported
boundary partitions or non-affine prescribed values are rejected. Geometry
generalization, arbitrary coefficients and non-affine boundaries remain open.

`losses/gnn.py` integrates the P1 relative L2 and gradient errors exactly on
triangles. The `sobolev` objective averages L2-squared plus `h1_weight` times
H1-seminorm-squared equally over graphs. It is verified against the independent
evaluation metrics and by gradient checking. The original `mse` objective remains
available. This is a supervised derivative loss; it does not implement a PDE
residual loss.

Training and checkpoint selection now use the declared objective; validation
history also records L2, H1 and learning rate. Deterministic epoch-based learning
rate decay is optional. Checkpoint schema 2 records architecture, objective,
gradient weight and decay configuration. Schema-1 checkpoints are not compatible
with this trainer; their archived predictions and manifests remain historical
evidence. Full and resumed schema-2 training are tested for exact equality.

The matched ablations use seed 0, 300 epochs, width 32, depth 3, batch size 7,
Adam .001 and a factor-.5 decay every 100 epochs. Batch size 7 divides 49 training
instances exactly. Both architecture choices are compared with both losses,
selecting exclusively on original n=8 validation data. The global model bundles
global communication and smooth lifting; this experiment does not isolate those
two architectural changes from each other.

```bash
.fenicsx-env/bin/python scripts/run_gnn_ablations.py \
  --dataset outputs/pre_gnn_20261028/square_dataset_extended/dataset_manifest.json \
  --normalization outputs/pre_gnn_20261028/square_audit_extended/normalization.json \
  --output outputs/my_repair_ablations
```

After a validation-only extension to 800 epochs, the fixed repair qualification
uses `configs/experiments/gnn_repair_qualification.json`. The parameter-only holdout
declaration is `gnn_repair_holdout_protocol.json`: 16 fresh test instances at n=8/24,
excluding all six previously examined test instances. Training/validation remain
49/9 at n=8, with unchanged normalization statistics and unchanged acceptance
thresholds. All three seeds train before any final test evaluation.

Generate the new dataset with `gnn_repair_dataset.json`, fit normalization only on
training n=8, then run `scripts/run_gnn_study.py` with the repair configuration.
Use fresh output directories. See [repair evidence](gnn_repairs.md) for measured
results and remaining limitations.
