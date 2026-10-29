# Reliability implementation and ordered deliverables

The October 28 increment addresses environment specifications, automated numerical
checks, checkpoint provenance, configurable evaluation, and multi-seed studies.
It preserves the CPU reference and the independent analytical/FEM/neural fields.
See [executed tests and 12-run study results](validation_20261028.md).

## Environment recreation

The portable environment specification remains `configs/environments/fenicsx.yml`.
Install the optional CPU backend explicitly after creating that environment:

```bash
python -m pip install -e '.[dev]'
python -m pip install -r configs/environments/torch_cpu.txt
python -c "import torch; assert torch.version.cuda is None"
```

The installed Linux x86-64 conda packages were captured in
`configs/environments/fenicsx-linux-64.explicit.txt`, including archive URLs and
MD5 checksums. The pip overlay is captured separately in
`configs/environments/torch_overlay_versions.txt`; `installed_versions.json`
records observed Python distribution versions. These are environment snapshots,
not evidence that a clean reinstall has passed. Conda builds may require the
recorded CPU instruction set and compatible Linux runtime.

```bash
micromamba create -p .reproduced-env -f configs/environments/fenicsx-linux-64.explicit.txt
micromamba run -p .reproduced-env python -m pip install -r configs/environments/torch_overlay_versions.txt
micromamba run -p .reproduced-env python -m pip install --no-deps --no-build-isolation -e .
micromamba run -p .reproduced-env python -m pytest -q
```

`.github/workflows/numerical.yml` defines base, CPU Torch and FEniCSx+Torch jobs.
It includes full tests and mesh convergence, plus standalone FEniCSx comparisons.
Explicit dependency imports prevent an accidentally missing optional package
from silently turning its required CI job into skipped tests. Workflow design
follows the [GitHub Python guide](https://docs.github.com/en/actions/tutorials/build-and-test-code/python)
and [setup-micromamba instructions](https://github.com/mamba-org/setup-micromamba).
Hosted workflow execution and a clean environment recreation remain unverified
in this workspace. The current directory has no Git metadata or configured remote.

## Checkpoint compatibility change

New checkpoints use schema **2**, with a SHA-256 fingerprint of physics, models,
losses, training and core Python modules. Both backends reject source changes
before continuing training. The fingerprint deliberately excludes plotting,
evaluation and tests; it works from installed package source as well as editable
source. Existing schema-1 checkpoints remain archived but cannot resume through
the strict loader. Start a new experiment rather than silently treating an old
checkpoint as scientifically compatible. A future explicit migration facility
must record origin and changed semantics separately.

## Evaluation and acceptance

`run_pinn.py` now accepts `--validation-seed`, `--holdout-points`,
`--boundary-points`, and `--evaluation-quadrature-order`. The default evaluation
seed is **10007**, independent of the training seed. Earlier non-default training
seeds implicitly changed the evaluation seed; manifests now expose it explicitly.

Optional `--acceptance` takes a JSON mapping of error metrics to maximum values.
Execution status remains separate from `acceptance.status` (`passed`, `failed`,
or `not_assessed`). A completed run need not satisfy the declared accuracy target.
Quadrature accumulation streams over triangle batches. Torch L2/H1 evaluation
computes first derivatives only; Laplacians are reserved for residual evaluation.
Nodal/boundary/holdout evaluation and the underlying mesh still occupy memory.

## Multi-seed study

```bash
python scripts/run_pinn_study.py --config configs/experiments/pinn_study_cpu.json \
  --output outputs/my_pinn_study
```

Use the CPU Torch environment for the supplied configuration. It declares two
cases, seeds 7/19/43, and 128/256 collocation points, for 12 separate training runs.
Acceptance is declared before execution: relative L2 <= 0.01, relative H1 <= 0.03,
maximum boundary error <= 1e-12, normalized held-out residual RMS <= 0.1.
These are initial study targets, not universal solver tolerances.

All runs share 4,096 evaluation points with seed 10007. Training points differ
across seeds. Study manifests group results by case and collocation count and
report median, minimum, maximum and population standard deviation for completed
runs, alongside execution failures and acceptance rate over **all attempts**.
Per-run FEM references, checkpoints, plots and manifests are preserved. Progress
is written after each run. An execution failure makes the CLI return nonzero;
accuracy rejection remains a separately reported scientific result.

This study measures robustness within the existing manufactured cases. It does
not establish held-out-instance generalization, confidence intervals, timing
superiority, or acceptance on unseen geometries. Keep a separate evaluation seed
for final assessment after selecting configurations.

## Ordered work remaining

1. Execute the hosted validation matrix and verify a clean environment recreation.
2. Review the initial multi-seed results; expand seeds and preserve a final assessment set.
3. Initial repeated cost studies, peak RSS and evaluation sensitivity are now
   implemented; see [scope and timing definitions](cost_study.md). Broader candidate
   grids and finer separation of optimizer setup/update costs remain open.
4. Square mixed-boundary PINNs with manufactured flux cases are now implemented;
   see [formulation and scope](mixed_pinn.md).
5. Domain sampling, callable coefficient gradients, constraint interfaces and a
   Dirichlet annulus PINN are implemented; additional irregular cases remain open.
6. Initial annulus PDE-instance datasets, whole-instance splits and graph invariants
   are implemented. Loading/batching and richer dataset families remain open.
7. Build and validate the supervised GNN baseline.
8. Execute cloud experiments, implement the viewer, and profile before GPU/kernel work.

No new PDE family, graph dependency or GPU runtime is introduced by this increment.
