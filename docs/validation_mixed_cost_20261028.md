# Mixed PINNs and CPU cost validation — 28 October 2026

## Executed checks

- Full installed-environment suite: **206 passed, 1 expected skip** (75.77 seconds).
- Lightweight suite: **165 passed, 18 expected skips** (35.41 seconds).
- The installed suite executed all **14 FEniCSx integration fixtures**.
- Four standalone new mixed-case comparisons passed (two cases, n=4 and n=8).
- Original Poisson mesh convergence completed for n=8,16,32.
- Both mixed-boundary training experiments completed and passed declared acceptance.
- All **30 isolated CPU cost trials** completed without execution failures.

## Mixed-boundary accuracy

CPU float64 Torch, width 8/depth 2, 256 collocation points, seed 7, Adam 2,000
steps, learning rate 0.003. Neumann loss uses 32 Gauss points and weight 1.
Evaluation uses n=16 (289 nodes/512 triangles), order 5, 4,096 held-out points
with seed 20007, and 257 dense edge midpoints for flux errors.

| Case | Relative L2 | Relative H1 seminorm | Relative Neumann flux RMS |
|---|---:|---:|---:|
| mixed_sine_right | 0.00021490041 | 0.0019252778 | 0.0013425927 |
| mixed_sine_left | 0.00014370313 | 0.001780078 | 0.0013988669 |

Dirichlet error was zero at evaluated points for both runs. The acceptance
limits were declared in `configs/experiments/pinn_mixed_cpu.json` before execution.
Largest external FEM nodal difference over the four comparisons: **8.88178e-15**.
The manufactured cases test both outward-normal orientations. These single-seed
runs do not establish robustness of mixed-boundary training across seeds.

## Repeated accuracy-versus-cost experiment

Hardware: AMD Ryzen 5 5600H with Radeon Graphics; CPU execution only.
Python 3.12.13; Torch 2.10.0+cpu; one requested Torch/BLAS/OpenMP thread.
Common targets: relative L2 <= 0.01 and relative H1 seminorm <= 0.06.
Three fresh-process repeats per candidate; same seed 7 for timing repeatability.
Each selected candidate passed both targets in every repeat.

| Case | Selected FEM mesh | FEM median seconds | Selected Torch steps | Torch median seconds |
|---|---:|---:|---:|---:|
| sine | fem_32 | 0.10731 | torch_300 | 1.90957 |
| variable_sine | fem_32 | 0.13431 | torch_300 | 1.85321 |

FEM cost includes mesh construction and assembly/solve. Torch cost includes
initialization, optimizer construction, updates and inference-module restoration.
Explicit module imports before the timer, process startup, checkpoint I/O and
error evaluation are excluded. Lazy optimizer/runtime initialization during
training remains included. These are common-tolerance comparisons among the
supplied candidates, not equal-error comparisons or universal optimal settings.

| Case | Candidate | Median pre-evaluation peak RSS (MiB) |
|---|---|---:|
| sine | fem_32 | 60.42 |
| sine | torch_300 | 305.46 |
| variable_sine | fem_32 | 60.01 |
| variable_sine | torch_300 | 306.20 |

RSS includes libraries and runtime state; it is not incremental model memory.
The largest neural L2/H1 variation across the three integration settings was 1.59e-14.
Inference samples, baseline RSS, every trial, and all evaluation settings are
preserved in the cost manifest. Numerical integration was stable for these
fields/settings; this is not a general guarantee for larger or sharper networks.

## Environment recreation and remaining gates

A clean-environment recreation was attempted with the explicit snapshot and a
writable temporary package cache. Required package archives were unavailable;
micromamba attempted downloads and failed to resolve conda.anaconda.org. This
gate is **blocked/unverified**, not passed. Hosted CI and actual cloud execution
also remain unverified. No user environment was replaced.

Irregular neural domains, general boundary/coefficient interfaces, PDE-instance
datasets, graph models, a viewer and GPU work remain open.

## Evidence

Source SHA-256: `cbea075b5b07733a14b61b116e648185346a23ad2b6f16fab836db7c758d80a4`. Git metadata remains unavailable.
Both mixed-run manifests and the cost study match the current source fingerprint.

[Cost manifest](../outputs/pending_20261028/cost/cost_manifest.json),
[right-edge run](../outputs/pending_20261028/mixed_sine_right/manifest.json),
[left-edge run](../outputs/pending_20261028/mixed_sine_left/manifest.json),
[FEniCSx comparisons](../outputs/pending_20261028/fenicsx/manifest.json),
[mesh convergence](../outputs/pending_20261028/convergence/convergence_manifest.json),
[environment attempt](../outputs/pending_20261028/environment_recreation_manifest.json).

Reproduction and mathematical scope: [mixed PINNs](mixed_pinn.md), [cost study](cost_study.md).
