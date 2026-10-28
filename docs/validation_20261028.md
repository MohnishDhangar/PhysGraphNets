# Reliability validation — 28 October 2026

Implemented: CPU environment specifications/snapshots, a hosted validation workflow,
checkpoint source compatibility, explicit evaluation settings and acceptance,
streamed neural quadrature, and controlled seed/collocation studies.

## Executed checks

- Installed environment: **192 passed, 1 expected skip** in 76.83 seconds.
- Lightweight environment: **156 passed, 13 expected skips** in 44.06 seconds.
- Original mesh convergence demo: completed for n=8,16,32; expected L2/H1 behavior retained.
- Study: **12 completed, 12 accepted, 0 execution failures**.
- Hosted CI and clean environment recreation have not been executed here.

The new tests exercise provenance mismatch/legacy rejection, affine quadrature
across batches, evaluation configuration, acceptance validation, failure accounting,
common evaluation seeds, and an actual two-seed experiment.

## Controlled study

Cases: sine and variable_sine. Seeds: 7,19,43. Collocation sizes: 128,256.
CPU float64 Torch, width 8/depth 2, Adam 1,000 steps at learning rate 0.003.
Evaluation: 16×16 mesh (289 nodes, 512 triangles), quadrature order 5,
4,096 common residual points with seed 10007, 257 points per boundary edge.
Thresholds were set in the config before execution: L2<=0.01, H1<=0.03,
boundary error<=1e-12, normalized held-out residual RMS<=0.1.

| Group | Median relative L2 | Worst relative L2 | Median relative H1 | Accepted |
|---|---:|---:|---:|---:|
| sine/n128 | 0.0045298927 | 0.0048224086 | 0.010586632 | 3/3 |
| sine/n256 | 0.0037890595 | 0.0048700397 | 0.010298191 | 3/3 |
| variable_sine/n128 | 0.00075864184 | 0.0007772702 | 0.0080882484 | 3/3 |
| variable_sine/n256 | 0.00070105388 | 0.00071831303 | 0.0080350477 | 3/3 |

Three seeds give an initial sensitivity check, not a reliable estimate of rare
failure probability. Increased collocation lowered median L2 in both cases, but
did not lower the worst sine L2. No matched-accuracy performance or instance
generalization claim follows from these runs.

## Evidence

Study source SHA-256: `c25fcf84cdb0efe557bd35d85f354dbe5008bc82394d42caee7d62c8dc4fbdc1`.
All 12 per-run manifests match this fingerprint. Git revision is unavailable.
The convergence manifest retains its own source snapshot from before the pip-overlay specification was added.

[Study manifest](../outputs/reliability_20261028/pinn_study/study_manifest.json) and
[convergence manifest](../outputs/reliability_20261028/convergence/convergence_manifest.json).
Generated outputs remain ignored. [Commands, compatibility changes and remaining gates](reliability.md).
