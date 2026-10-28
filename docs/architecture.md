# Architecture — Phase 0 design freeze

## Pipeline

`ManufacturedCase -> Mesh2D -> P1 FEM assemble/solve -> FieldResult -> .npz + metrics + 2D rendering`

All packages use a single `src/physgraph/` implementation. Cloud notebooks,
scripts and tests import this package. The small verification case is solvable
without PyTorch or CUDA.

## Packages in Phase 1

- `core/types.py`: shared contracts for mesh, PDE case, result.
- `mesh/structured.py`: unit-square triangular mesh generation.
- `physics/manufactured.py`: exact field, forcing, exact gradient.
- `fem/poisson.py`: P1 element matrices, 3-point quadrature, Dirichlet solve.
- `fem/error.py`: element-quadrature L2/H1 error estimates.
- `experiments/pipeline.py`: evaluate/run one case or convergence study.
- `io/results.py`: versioned NPZ format without pickle.
- `visualization/plotting.py`: headless PNG field and convergence figures.

## Neural and deferred packages

- `graph/`: mesh-to-graph map, edge/node feature definitions, sparse formats.
- `models/`: initial CPU PINN implemented; MeshGraphNet remains deferred.
- `losses/`: strong-form PINN residual implemented; graph weak-form/energy losses deferred.
- `training/`: deterministic PINN Adam/checkpoint/resume implemented; graph datasets/batches deferred.
- `profiling/`: timing and peak memory baselines; CUDA candidate selection.
- `cuda/`: an implementation only after profiling proves its value.
- `viewer/`: interactive renderer, field probes, colormaps, domain controls.

## Phase 2A extension

`EllipticProblem + Mesh2D -> assemble A/M/F -> ReducedSystem -> CPU solve`.
`AnalyticalReference` is supplied separately to evaluation, never to assembly.
`physics/base.py`, `boundary.py`, and `diffusion.py` define physical inputs and
manufactured cases. `mesh/nonuniform.py` builds graded conforming triangles.
The FEM extension separates `geometry.py`, `quadrature.py`, `boundary.py`,
`assembly.py`, `operators.py`, and `elliptic.py`. Reference Poisson assembly
remains available for independent comparisons. Error integration is shared.
`experiments/elliptic.py` and `experiments/smoke.py` contain orchestration and
checks; scripts only parse arguments and call package functions.
`io/manifest.py` records provenance for both phases. See `phase2.md` for details.

The annulus extension adds `mesh/annulus.py`, `physics/annulus.py`, and
`experiments/annulus.py`. The existing elliptic assembler consumes named exterior
edge groups through `BoundaryConditions.dirichlet_tags`. Tagged meshes use field
schema v2. Plots use the supplied triangles, preserving holes. See `annulus.md`.

## Cloud portability contract

No cloud-specific numerical source files. Notebook bootstrap downloads or
extracts the identical source tree; runs the same tests and command-line files.
GPU hardware is introduced as an experiment parameter, not solver architecture.

The rectangle/cylinder extension uses `mesh/cylinder.py`, `physics/cylinder.py`,
and `experiments/cylinder.py`, reusing the elliptic solver and schema-v2 exports.
See `cylinder.md` for the formulation and geometry contract.

The airfoil extension uses `mesh/airfoil.py`, `physics/airfoil.py`, and
`experiments/airfoil.py`, reusing polynomial manufactured fields and the existing
elliptic solver. The exact closed-profile convention and limits are in `airfoil.md`.

The initial PINN separates `models/pinn.py`, `physics/pinn.py`, `losses/pinn.py`,
`training/pinn.py`, `evaluation/pinn.py`, `visualization/pinn.py` and
`experiments/pinn.py`. Independent comparisons live in `verification/` with lazy
FEniCSx imports. See `pinn.md` and `fenicsx.md`; independent verification passed
for the fixture set documented in `validation_20261026.md`.

The optional PyTorch backend follows the same separation in each layer's
`torch_pinn.py`. `core/torch_runtime.py` scopes CPU determinism and threading;
training shares the validated NPZ state format, and evaluation shares quadrature
and metrics while supplying backend-specific derivatives. `--backend torch`
loads Torch lazily, preserving the dependency-light NumPy and FEM paths.
