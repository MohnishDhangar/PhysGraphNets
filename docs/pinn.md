# Phase 2B: initial CPU strong-form PINN

The user authorized beginning PINN development alongside the FEniCSx comparison.
This records the Phase 2B decision. External FEM verification subsequently
passed for the tested fixture set; actual cloud execution remains open.
FEniCSx and CPU PyTorch 2.10.0 are now installed in `.fenicsx-env`, and the user
verified second-order PyTorch autodiff. The neural runner now offers both NumPy
and PyTorch CPU backends, with the same independent SciPy FEM reference.
See [the validation record](validation_20261026.md).

## Mathematical and implementation scope

The initial PINN fits **one PDE instance** on [0,1]². It is not an operator model
trained across instances. Supported cases are `sine`, `affine`, `variable_sine`,
and `affine_diffusion`, with the existing analytical forcing and references.
Two mixed-boundary square cases are also available: `mixed_sine_right` and
`mixed_sine_left`. They add a prescribed outward flux on one vertical edge.
See [mixed formulation and validation scope](mixed_pinn.md). A first curved-domain
case, `annulus_log`, is documented in [annulus PINN](annulus_pinn.md).

For -div(k grad(u))+c*u=f, the pointwise residual is

\[
r_\theta=-k\Delta u_\theta-\nabla k\cdot\nabla u_\theta+cu_\theta-f.
\]

In the variable-coefficient cases k=1+x+y, so grad(k)=(1,1). Omitting this term
would change the PDE; tests compare the implementation to independently
differenced fluxes. Coefficients/forcing come from `physics/`; learned inference
and derivatives are in `models/`; residuals are in `losses/`; training,
evaluation, plotting and CLI orchestration remain separate modules.

The model is a tanh MLP with input coordinates rescaled to [-1,1]², a linear
output, and default two hidden layers of width 8 (105 trainable parameters).
Trainable weights use seeded variance-scaled normal initialization and biases
start at zero. The prediction is

\[
u_\theta(x,y)=b(x,y)+16x(1-x)y(1-y)N_\theta(x,y).
\]

The affine lifting b is 0 for `sine`, 1+2x-3y for `affine`/`affine_diffusion`,
and 1+x+2y for `variable_sine`. These liftings encode boundary data only;
the interior sine field is never supplied to training. The factor enforces
Dirichlet data exactly in real arithmetic, including nonzero boundary values.
Boundary metrics allow roundoff from the analytical NumPy reference, e.g. sin(pi).

## Derivatives and optimizer

In the default NumPy backend, first and diagonal-second spatial derivatives are propagated by explicit chain
rules through the tanh MLP and the boundary mask. No spatial finite differences
are used in training. For tanh z, the first derivative is 1-tanh²z and the second
is -2*tanh(z)*(1-tanh²z).

The full-batch objective on fixed independently sampled collocation points is
mean(r²)/max(mean(f²),1). Parameter derivatives use complex-step differentiation
with step 1e-30. All operations in that path retain complex dtype, and the loss
uses r*r rather than conjugating or taking absolute values. This analytic
extension avoids subtractive finite-difference cancellation. Spatial derivatives
and parameter gradients are checked against independent central differences.
See the [SciPy discussion of complex-step Jacobians](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)
for the analyticity requirement; our optimizer implements the same derivative
principle directly, rather than calling SciPy least-squares.

Full-batch Adam uses beta1=0.9, beta2=0.999, epsilon=1e-8 and default learning
rate 0.003. The default is 128 fixed collocation points and 1,000 steps, seed 7.
Model/optimizer state is float64; derivative work uses complex128. Cost grows
linearly with the number of parameters per gradient, so this backend is intended
for small CPU verification experiments, not large-scale training or GPU claims.

The optional `--backend torch` implementation uses a `torch.nn.Module` with the
same tanh layers, flat weight layout, initial values and collocation points.
PyTorch differentiates the **complete constrained field**, including the lifting
and boundary mask. First spatial derivatives retain their graph for second
derivatives; training retains the second-derivative graph for parameter
backpropagation. See [`torch.autograd.grad`](https://docs.pytorch.org/docs/2.10/generated/torch.autograd.grad.html).
The residual includes the explicit `grad(k)` term above; the sampled physical
coefficients and forcing carry no analytical solution data into training.

Native [`torch.optim.Adam`](https://docs.pytorch.org/docs/2.10/generated/torch.optim.Adam.html)
uses the same hyperparameters, with `foreach=False`, `fused=False`, no weight
decay, and float64 parameters, moments and step counter. All tensors are on CPU;
there is no device selector. Default `--torch-threads 1` and deterministic
algorithms are scoped to training/evaluation and restored afterward. NumPy's
local generator supplies initialization and points; no global NumPy or Torch
random state is consumed. Tests compare values, first/second spatial derivatives,
residuals and parameter gradients against the independent NumPy implementation
on every supported case, and compare short Adam trajectories.

## Checkpoints and reproducibility

October 28 update: new checkpoints use schema 2 and require matching training
source fingerprints. Schema-1 checkpoints are not silently resumed. Evaluation
seeds/settings and numerical acceptance are now explicit, and a multi-seed runner
is available. See [reliability details and compatibility changes](reliability.md).

Checkpoints are compressed NPZ, loaded with allow_pickle=False. They contain:

- Parameters, Adam first/second moments, and completed step count.
- Exact collocation points, random-generator state, and full loss history.
- Training configuration, optimizer constants, backend and schema version.

Writes replace a temporary checkpoint atomically. Resume rejects mismatched
configurations, invalid/nonfinite states, and target steps earlier than the
saved step. `--steps` is the **total target step**, not additional steps. Tests
verify bit-identical complete parameters, optimizer state and loss history for
uninterrupted versus interrupted/resumed training in the same environment.
Identical behavior across different BLAS, NumPy, architectures or versions is
not asserted. Manifests record relevant environment and thread settings.
PyTorch checkpoints additionally record Torch version and CPU runtime settings;
resume requires them to match. Cross-backend resume is rejected. Both backends
use NPZ rather than pickle or `torch.load`. Reproducibility is limited to the
same software and hardware environment, consistent with
[PyTorch's reproducibility guidance](https://docs.pytorch.org/docs/2.10/notes/randomness.html).

## Evaluation and artifacts

The unchanged CPU FEM solver is run separately on the evaluation mesh. Analytical,
FEM and PINN arrays remain distinct. Direct evaluation of the neural value and
gradient at element quadrature points gives continuous L2/H1 metrics; these are
not errors of a piecewise-linear interpolation of the model's nodal output.
Nodal predictions are only used for visualization and nodal error measurements.

A separate 1,024-point random sample, seeded independently from training, measures
held-out spatial strong residual. It is a spatial check on the same PDE instance,
not held-out-instance generalization. Boundary checks use dense samples on all
four edges. Training loss alone is never treated as solution accuracy.

Runs save `checkpoint.npz`, `training.csv`, `training.png`, `comparison.png`,
`comparison.npz`, `fem_reference.npz`, `metrics.json`, and `manifest.json`.
The comparison archive uses explicit `exact`, `fem_solution`, `pinn_prediction`
keys. The existing field archive contains only the true FEM solution.
Manifests include errors, mesh sizes, training and FEM timings separately,
configuration, seeds, resume checkpoint digest, Python/dependencies, precision,
source digest and revision when available. `status: completed` means the run
finished; it is not a universal accuracy acceptance claim.

## Running

```bash
OPENBLAS_NUM_THREADS=1 python scripts/run_pinn.py --config configs/experiments/pinn_local.json
OPENBLAS_NUM_THREADS=1 python scripts/run_pinn.py --case variable_sine --steps 1000
OPENBLAS_NUM_THREADS=1 python scripts/run_pinn.py --case affine --steps 300
```

Use `.pignn-venv/bin/python` on this workstation. Output defaults to unique
folders under `outputs/pinn/`; explicit output folders must be empty.
To resume the default configuration to a total of 1,500 steps:

```bash
OPENBLAS_NUM_THREADS=1 python scripts/run_pinn.py --config configs/experiments/pinn_local.json \
  --resume outputs/pinn/<previous-run>/checkpoint.npz --steps 1500
```

For PyTorch, use the installed CPU build in `.fenicsx-env`:

```bash
"$HOME/.local/bin/micromamba" run -p "$PWD/.fenicsx-env" \
  python scripts/run_pinn.py --config configs/experiments/pinn_torch_cpu.json
"$HOME/.local/bin/micromamba" run -p "$PWD/.fenicsx-env" \
  python scripts/run_pinn.py --backend torch --case variable_sine --steps 1000
"$HOME/.local/bin/micromamba" run -p "$PWD/.fenicsx-env" \
  python scripts/run_pinn.py --config configs/experiments/pinn_torch_cpu.json \
  --resume outputs/pinn/<previous-torch-run>/checkpoint.npz --steps 1500
```

If a restricted shell cannot write micromamba's user cache, prefix the command
with `XDG_CACHE_HOME=/tmp/physgraph-cache`. `MPLCONFIGDIR=/tmp/physgraph-mpl`
similarly redirects plotting cache writes. This does not change the environment.

CLI values override JSON values. All training settings must match the checkpoint;
evaluation mesh size and output destination can change. PyTorch remains optional:
the default NumPy backend and FEM package work without it. Selecting Torch without
the dependency produces an explicit error before creating an experiment folder.
The tested Torch build is 2.10.0+cpu; no new environment installation is needed here.

## Remaining work

- Extend independent FEniCSx checks whenever new physics or fixtures are added.
- Study multiple seeds, collocation sensitivity and fair PINN/FEM cost comparisons.
- Extend neural boundary handling and domain sampling beyond the implemented square cases.
- Execute the same package on actual cloud hardware.
- Later, develop datasets and graph models for whole-instance generalization.

## PyTorch validation

The installed-environment suite passed **187 tests**, with one intentional
missing-dependency skip; all FEniCSx integration cases executed. Both PINN
backends passed 46 focused tests. Four single-seed Torch runs completed with
relative neural L2 errors of 0.004529893 (`sine`), 0.0007772702 (`variable_sine`),
0.0001868288 (`affine`) and 0.0001967195 (`affine_diffusion`). See the
[27 October validation record](validation_20261027.md) for configuration,
independent errors, limitations, commands and saved manifests.

## Initial NumPy validation (before FEniCSx installation)

The initial local suite passed **150 tests**, with **12 FEniCSx integration
tests skipped** because the external environment is absent. The original
Poisson mesh convergence demo also passed. Checkpoint resume was tested
both for bit-identical optimizer replay and through the experiment CLI.

The following single-seed runs used Python 3.14.7, CPU float64 parameters,
complex128 derivative work, width 8/depth 2 (105 parameters), 128 fixed
collocation points, learning rate 0.003, and seed 7. Evaluation used a 16×16
mesh (289 nodes, 512 triangles) and independent 1,024-point residual sampling.

| Case | Steps | Relative neural L2 | Relative neural H1 | Initial → final loss |
|---|---:|---:|---:|---:|
| `sine` | 1000 | 0.00452989 | 0.011119 | 1.0385 → 0.00139118 |
| `variable_sine` | 1000 | 0.00077727 | 0.00870199 | 1.00583 → 0.00128104 |
| `affine` | 300 | 0.000186829 | 0.00052261 | 1.55374 → 0.000676687 |

Boundary errors were at most 1.23e-16. These are reproducible starting points,
not cross-instance, multi-seed or performance claims. The affine FEM solution
remains exact to roundoff while the trained neural approximation has nonzero
error. Training takes substantially longer than these small FEM solves.

Per-run manifests in `outputs/pinn_validation/sine/` and
`outputs/phase2b_validation/` retain complete dependency versions, errors,
source hashes and timings. The aggregate record is
`outputs/phase2b_validation/validation_manifest.json`. Git revision was
unavailable in this workspace; source SHA-256 provenance is recorded instead.
