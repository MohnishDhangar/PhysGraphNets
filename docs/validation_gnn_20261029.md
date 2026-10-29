# CPU GNN development evidence, 29 October 2026

The supervised model and epoch-based trainer are implemented. Model tests cover
permutation equivariance, disjoint-graph isolation, boundary enforcement and
finite gradients. The trainer test compares two uninterrupted epochs with one
plus a resumed epoch, checking exact model and Adam-state equality.

A three-epoch development smoke run used the existing 49 training and 9 validation
instances at n=8. Validation area-weighted MSE was 0.117578, 0.0401414, 0.0378989.
These are supervised loss values, not relative L2 percentages. This short run
is not an accepted scientific benchmark and did not inspect test predictions.

Artifacts: `outputs/gnn_20261029/smoke/manifest.json`, `checkpoint.pt` and
`outputs/gnn_20261029/convergence/convergence_manifest.json`. The original
8/16/32 Poisson convergence demo completed with its established errors.

Still required: controlled multi-seed accuracy evaluation, declared model
acceptance thresholds, held-out-instance/resolution tests, inference/reporting
CLI, broader geometry training and profiling. Resume is at epoch boundaries;
mid-epoch recovery and a full experiment-config CLI remain future improvements.

Full installed CPU Torch/FEniCSx regression: **267 passed, 1 expected skip**,
executed in 83.08 seconds. No GNN test-accuracy acceptance is claimed.
