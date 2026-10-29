# Pre-GNN validation — 28 October 2026

The CPU graph-data prerequisites are implemented and validated. No GNN has been
trained. The next model/trainer and resume gates are listed in
[the entry protocol](gnn_protocol.md); unrelated geometry PINN, cloud, CI and
viewer objectives remain open in [the checklist](checklist.md).

## Executed checks

| Check | Result |
| --- | --- |
| Full CPU Torch/FEniCSx suite | 265 passed, 1 expected missing-dependency-path skip |
| Lightweight NumPy/SciPy environment | 218 passed, 24 optional Torch/FEniCSx skips |
| New square fixtures against FEniCSx, n=4/8 | Four comparisons passed; max nodal difference 2.949e-13 |
| Original Poisson convergence, n=8/16/32 | L2 0.0422122 → 0.0107514 → 0.00270066; H1 0.194378 → 0.0979258 → 0.0490562 |
| New square family, both boundary modes | L2 rates 1.948–1.990; H1 rates 0.979–0.996 |
| New family assembly order 7 versus 9 | Maximum nodal difference 4.885e-15 |
| New family error order 5 versus 7 | Maximum reported-error difference 1.596e-11 |
| Annulus schema-2 compatibility | All 32 regenerated archives loaded and batches validated |
| Offline wheel build | Passed in `.fenicsx-env`; package imports/data loading/batching checked from the wheel in both environments |

The installed environment uses Python 3.12.13 and CPU PyTorch 2.10.0+cpu.
The lightweight environment uses Python 3.14.7. Tests exercise malformed
manifests/archives, checksum failures, topology/normal/physics corruption,
split consistency, nonzero prescribed values, input/target separation,
training-only normalization, reserved resolutions and batch reconstruction.
The full suite includes all 16 independent FEniCSx fixtures.

## Dataset and readiness result

The initial 32-instance design generated only 29 training, 2 validation and 1
test instance with split seed 0. Its readiness audit **failed** the predeclared
minimum instance counts. It remains in `square_dataset/` and
`square_audit_initial/`; no split seed or acceptance threshold was changed.

The design was extended to the next 32 parameter settings before model training,
preserving every previous identity and assignment. The resulting **64 instances,
128 FEM-labelled graphs** contain 49 training, 9 validation and 6 test instances.
Each has n=8 and n=16 meshes, respectively 81/289 nodes and 128/512 triangles.
All 128 archives passed schema checks, physical-feature reconstruction,
prescribed-value checks, FEM residual reassembly and recomputed analytical errors.

| Label-quality metric, maximum over all graphs | Observed | Required |
| --- | ---: | ---: |
| Relative reduced FEM residual | 2.182e-15 | <=1e-10 |
| Relative residual after higher-order reassembly | 2.504e-15 | <=1e-7 |
| Relative analytical L2 error | 0.0197927 | <=0.05 |
| Relative analytical H1-seminorm error | 0.268082 | <=0.35 |

These measure FEM discretization quality, not neural accuracy. Every model input
is independent of solved labels. The saved normalizer uses only the 49 training
instances at n=8. The n=16 meshes are reserved before training; validation model
selection must likewise use n=8. The readiness audit checks physical label quality
at both resolutions but does not select a model using held-out predictions.

The regenerated annulus example remains an auxiliary geometry/contract dataset:
it has only one validation instance and is not designated the primary training
dataset. Its earlier artifacts were preserved.

## Artifacts and provenance

Generated artifacts are under `outputs/pre_gnn_20261028/` and remain ignored:

- `square_dataset_extended/dataset_manifest.json`: physical identities, splits,
  checksums, mesh counts, FEM errors and generator provenance.
- `square_audit_extended/manifest.json` and `normalization.json`: passing gates,
  reserved resolutions and training-only fit provenance.
- `family_verification/manifest.json`: convergence and quadrature sensitivity.
- `fenicsx/manifest.json`: independent assembly, load and solution parity.
- `convergence/convergence_manifest.json`: original Poisson regression.
- `annulus_schema2_contract_manifest.json`: compatibility evidence.
- `build_manifest.json` and `dist/physgraph-0.1.0-py3-none-any.whl`: package build,
  wheel checksum and separate Python 3.12/3.14 import/data/tensor checks.

The reference revision was `1b1e773a417d9b520482e2283ef7057d985a30ef` with local
changes. The final source/config fingerprint for expanded generation and audit is
`0801b1112d12644493f86d7e67ac6ad914a5249fd110b938b93378d2f8a24a97`.
Earlier family verification used fingerprint
`4363fe0f34306e15da8ca7b95d60aee56cf8878a119fb1677f3e364e09d6a4b2`;
the subsequent change expanded the dataset config from 32 to 64 instances.
Each experiment records its actual source fingerprint rather than rewriting
earlier evidence.

The first wheel attempt in `.pignn-venv` could not import `setuptools.build_meta`.
The installed `.fenicsx-env` had setuptools 84.2.1 and wheel 0.48.0, so the wheel
was built there with `--no-index --no-deps --no-build-isolation`. No environment
reinstall or new runtime dependency was needed.

## Reproduction

Run the commands in [graph_data.md](graph_data.md) with the installed environment
and fresh output paths. The numerical suite and independent adapter were run with
`OPENBLAS_NUM_THREADS=1`; this sandbox also needs `UCX_TLS=self` for single-process
MPI and writable temporary caches:

```bash
env XDG_CACHE_HOME=/tmp/physgraph-cache MPLCONFIGDIR=/tmp/physgraph-mpl \
  OPENBLAS_NUM_THREADS=1 UCX_TLS=self \
  "$HOME/.local/bin/micromamba" run -p "$PWD/.fenicsx-env" \
  python -m pytest -q
```

No hosted CI or cloud-hardware execution is claimed. No inference speedup,
operator-learning accuracy or geometry extrapolation result exists yet.
