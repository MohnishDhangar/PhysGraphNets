# PhysGraph — CPU FEM reference and initial PINN baseline

Latest increment: [reproducibility and controlled PINN studies](docs/reliability.md),
with [28 October validation results](docs/validation_20261028.md).

This is a **working numerical foundation with an initial CPU PINN baseline**.
Graph models remain future work. It solves
Poisson and scalar variable-coefficient diffusion/reaction equations with
Dirichlet or mixed Dirichlet/Neumann boundaries using linear triangular (P1)
finite elements. Uniform and graded unit-square meshes and six manufactured
cases provide analytical verification. A tagged radial/angular annulus adds a
logarithmic diffusion reference with Dirichlet and mixed boundaries. Runs save
metrics, mesh-aligned fields, visual comparisons and environment manifests.
See [Phase 2A](docs/phase2.md) and the [annulus milestone](docs/annulus.md).

## Target development strategy

- **Local Arch Linux (Ryzen 5, 8 GB RAM):** CPU FEM, correctness tests, coding,
  field visualization. NVIDIA CUDA is not required for Phase 1.
- **Kaggle GPU / Google Colab:** the *same repository*, initially CPU smoke tests;
  later GPU training of PINN/GNN. Notebook entry point is provided.
- **CUDA kernels:** deliberately absent until the ordinary model has passed
  numerical validation and memory/latency profiling.

## Local quick start (Fish shell)

From the repository root (the folder containing this README):

```fish
python -m venv .venv
source .venv/bin/activate.fish
python -m pip install -e '.[dev]'
python -m pytest -q
python scripts/run_phase1.py --config configs/experiments/poisson_local.json
python scripts/mesh_convergence.py --sizes 8 16 32
python scripts/run_phase2.py --config configs/experiments/elliptic_local.json
python scripts/run_annulus.py --config configs/experiments/annulus_local.json
```

`python -m pip` is generally reliable inside a virtual environment, even if
`pip` is not on your shell's PATH. No `sudo pip` required.

**Expected files:**

- `outputs/phase1_local/comparison.png` — solution, analytical field, nodal
  error, mesh side by side.
- `outputs/phase1_local/field.npz` — portable mesh/field archive.
- `outputs/phase1_local/metrics.json` — residual and numerical errors.
- `outputs/convergence/convergence.csv`, `.png` — FEM mesh refinement study.
- `outputs/phase2/<unique-run>/` — elliptic fields, plots, metrics, convergence
  rates and a manifest containing environment/configuration/source provenance.

Try a nonzero boundary case:

```fish
python scripts/run_phase1.py --case affine --nx 20 --ny 20 --output outputs/affine
```

Change the sample case to `sine`, `polynomial`, or `affine`.

For Phase 2A, use `variable_sine`, `affine_diffusion`, or `mixed_polynomial`:

```fish
python scripts/run_phase2.py --case mixed_polynomial --grading 1.3 --convergence 8 16 32
```

Phase 2A creates unique output directories by default. Explicit `--output`
directories must be empty. The existing environment on this workstation is
`.pignn-venv`; activate it instead of creating `.venv` if desired.

For the cylinder cross-section geometry milestone:

```fish
python scripts/run_annulus.py --boundary-mode inner_neumann --convergence 4 8 16
```

This solves scalar diffusion on a straight-sided approximation of an annulus.
It saves tagged fields, plots, solution/geometry convergence and manifests under
`outputs/annulus/`. The airfoil geometry milestone is also available (see `docs/airfoil.md`); aerodynamic flow remains future work.

## On Kaggle / Colab

Two approaches:

1. Push this repository to GitHub and clone it in your notebook (when internet
   access is available), or
2. Upload the project ZIP as a Kaggle dataset/notebook input or upload it to
   Colab. Use `notebooks/cloud/phase1_cloud.ipynb` as a guide.

Within the unpacked project, the same commands run:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python scripts/cloud_smoke.py
python scripts/run_phase1.py --config configs/experiments/poisson_cloud.json
python scripts/run_phase2.py --config configs/experiments/elliptic_cloud.json
```

Phase 1 does **not use GPU intentionally**. Save or download the `outputs/`
folder before a free-tier notebook session terminates. In Kaggle, persist
results through saved notebook outputs or a separately managed dataset.

## What is numerically implemented?

- Uniform unit-square grids split into CCW P1 triangles.
- Quadrature-based finite-element RHS and exact P1 stiffness assembly.
- Sparse Dirichlet elimination (including nonzero boundary values).
- Independent exact solutions for three manufactured PDE cases.
- Relative continuous-style L2 and H1-seminorm error estimates using triangle
  quadrature; maximum nodal error and reduced-system residual.
- Mesh refinement benchmark, field archive, and PNG visualization.

## What is intentionally not implemented yet?

Gmsh imports and arbitrary-domain mesh generation, pure Neumann and Robin
boundaries, tensor diffusion, irregular-domain PINNs, GNN training,
time-dependent PDEs, a live viewer, GPU profiling, and custom CUDA kernels.
Remaining milestones are documented in `docs/roadmap.md`.

## Scientific caveats

- The plot uses interpolated **nodal** error, while the L2/H1 metrics use
  element quadrature and compare fields/gradients inside triangles.
- Phase 1 assembly uses three-point quadrature. Phase 2A uses higher-order
  triangle and edge quadrature, and error integration uses an independent
  higher-order rule in both phases. Refine meshes and check quadrature-order
  sensitivity for new coefficient/forcing fields.
- `relative_residual` measures the **interior finite-element linear solve**;
  a near-zero residual does not mean zero error from the continuum PDE.
- This example is a Poisson FEM **reference**, not an independently trained model.

See `docs/architecture.md`, `docs/data_contract.md`, `docs/verification.md`
and `AGENTS.md` before adding future modules.

## Rectangle with a cylinder

The next CPU geometry milestone is implemented: an exact rectangular outer
boundary with a tagged polygonal cylinder obstacle, verified using manufactured
diffusion/reaction fields. See [formulation and validation](docs/cylinder.md).

```bash
python scripts/run_cylinder.py --config configs/experiments/cylinder_local.json
```

Outputs include fields, plots, convergence results, and provenance manifests.
Independent FEniCSx checks passed for this fixture set; a square-domain CPU PINN is also available.

## Symmetric airfoil exterior

A closed symmetric NACA-style profile inside a circular far field now supports
verified scalar diffusion/reaction experiments, tagged upper/lower surfaces,
rotation and scaling, and separate solution and geometry convergence metrics.
See [profile convention, formulation, and limitations](docs/airfoil.md).

```bash
python scripts/run_airfoil.py --config configs/experiments/airfoil_local.json
```

This milestone verifies elliptic FEM on airfoil geometry. Flow physics,
airfoil PINN training, and graph models remain future work.

## Independent comparison and initial PINN

```bash
python scripts/compare_fenicsx.py --resolutions 4 8
OPENBLAS_NUM_THREADS=1 python scripts/run_pinn.py --config configs/experiments/pinn_local.json
```

The FEniCSx comparison requires a separate DOLFINx/PETSc environment and exits
nonzero with an explicit blocked manifest if dependencies are missing. Its
12 integration cases and 24 comparison runs passed in `.fenicsx-env`; the full
suite passed 161 tests with one expected skip. See [validation evidence](docs/validation_20261026.md)
and [setup and scope](docs/fenicsx.md).

The PINN runs with the existing NumPy/SciPy environment: a small tanh MLP,
analytical spatial derivatives, complex-step parameter gradients, hard square
Dirichlet conditions, and deterministic checkpoint/resume. It preserves separate
analytical, FEM, and neural fields. See [mathematics, usage and limits](docs/pinn.md).

The optional PyTorch backend adds CPU float64 automatic differentiation and native
Adam for the same four square Dirichlet cases. It includes derivative checks
against NumPy, complete NPZ checkpoint/resume, direct continuous neural errors,
and separate FEM reference artifacts. Run it in the installed environment:

```bash
"$HOME/.local/bin/micromamba" run -p "$PWD/.fenicsx-env" \
  python scripts/run_pinn.py --config configs/experiments/pinn_torch_cpu.json
```

`--backend numpy` remains the default; PyTorch is optional for the base package.
The initial PyTorch suite passed **187 tests** with one expected skip, and four
PyTorch case runs completed. See [PyTorch validation evidence](docs/validation_20261027.md).

Subsequent capabilities include [mixed-boundary square PINNs](docs/mixed_pinn.md)
with independent FEniCSx checks and [repeated CPU accuracy-versus-cost studies](docs/cost_study.md).
The latest numerical suite passed **206 tests** with one expected skip.
See [mixed-boundary and cost-study results](docs/validation_mixed_cost_20261028.md).
