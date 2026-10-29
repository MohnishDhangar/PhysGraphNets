# CPU GNN entry protocol

The data gates, CPU model and trainer are implemented. The first controlled
three-seed qualification failed its accuracy criteria; see
[results](validation_gnn_flow_20261029.md). The protocol below remains the data
separation rule. No CUDA, Triton, PyTorch Geometric or DGL dependency has been
added; training uses ordinary CPU PyTorch tensor operations.

## Frozen data usage

Use the 64-instance parameterized-square config with split seed 0. Physical
identities determine 49 training, 9 validation and 6 test instances. All mesh
refinements inherit their physical instance's split. Never split nodes or assign
two refinements of the same physical instance to different instance splits.

- Fit input normalization and optimize weights using training instances at n=8.
- Select hyperparameters/checkpoints using validation instances at n=8 only.
- Keep n=16 out of preprocessing, optimizer updates and checkpoint selection.
- After choices are frozen, report test instances at n=8 and n=16 separately.
- Report training/validation instances at n=16 separately as a resolution
  diagnostic; do not describe these as unseen physical instances.

The readiness audit examines held-out **FEM label quality** before training; it
does not use a learned prediction or select a model. Test labels are still
excluded from model selection. The saved normalizer records exactly which
training archive checksums and resolutions contributed to its statistics.
The default training batch iterator follows those resolutions. Explicitly pass
`resolutions=[8]` for validation used during model selection; the general
validation/test iterator otherwise returns all available resolutions.

Hash splitting removes exact physical duplicates across splits; it does not
guarantee parameter extrapolation. This small test set supports an initial
baseline only. New families or geometry extrapolation need separately declared
datasets and split rules, not a stronger claim about this one.

## Numerical entry gates

The checked-in readiness config requires at least 8/3/3 train/validation/test
physical instances, at least two resolutions, valid positive triangles, complete
boundary/topology features, finite FEM targets and exact prescribed values.
Every label must satisfy relative algebraic residual <=1e-10, relative analytical
L2 error <=0.05 and H1-seminorm error <=0.35. Reassembling with two additional
quadrature orders must give relative residual <=1e-7. These are **label quality
gates**, not GNN accuracy targets. The case fixtures also require approximately
second-order L2 and first-order H1 convergence and independent FEniCSx parity.

## Deliverables after the entry gates

1. Implement a supervised CPU float64 encoder/message-passing/decoder model
   consuming only documented node/edge features. Apply exact Dirichlet values
   using the raw mask/value arrays; all interior and Neumann-node values remain
   predictions. Keep physical parameters, FEM targets and analytical references
   outside the model input interface unless explicitly represented as documented
   physical features.
2. Implement deterministic optimization and checkpoints containing model,
   optimizer, RNG, epoch/batch progress, source compatibility, dataset checksum,
   split protocol and normalization provenance. Demonstrate uninterrupted versus
   resumed equivalence before a training study.
3. Declare architecture, optimization settings, seeds, stopping rule and model
   acceptance thresholds before studying test predictions. Select on validation
   only. Record failures and all seeds; no selecting the best test seed.
4. Report separate FEM-relative P1 L2/H1, physical boundary errors and assembled
   residuals using `evaluation.graph.prediction_metrics`. Aggregate per-instance
   scores at each resolution with equal instance weight, not pooled node counts.
   Also report spread/worst cases; analytical error remains distinct from FEM
   label error. Include simple prescribed-boundary baselines and FEM cost.
5. Measure training cost and repeated inference separately on the same CPU and
   precision, with warm-up/timing and memory methodology. Expand geometry and
   add physics losses only after the supervised baseline and resume gates pass.

Additional annulus mixed PINNs, cylinder/airfoil neural constraints, hosted CI,
cloud execution and the viewer remain independent roadmap objectives. They are
not prerequisites for the first supervised square-domain GNN baseline.
