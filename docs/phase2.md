# Phase 2A: variable-coefficient elliptic FEM

The CPU numerical part of Phase 2 is implemented. An initial CPU PINN is now implemented in [Phase 2B](pinn.md). The independent
[FEniCSx comparison](fenicsx.md) passed for the documented fixtures and resolutions. This step
adds no dependencies. Local and cloud runners import the same Python package.

## Mathematical problem

On a triangulated domain, solve

\[
-\nabla\cdot(k\nabla u)+cu=f,\qquad k>0,\quad c\ge0,
\]

with prescribed \(u=g\) on a nonempty Dirichlet boundary \(\Gamma_D\), and
\(k\nabla u\cdot n=q\) on the remaining Neumann boundary \(\Gamma_N\).
Here \(n\) is the outward unit normal. The physical outward flux is **-q**.
For test functions vanishing on \(\Gamma_D\),

\[
\int_\Omega k\nabla u\cdot\nabla v+cu v
=\int_\Omega f v+\int_{\Gamma_N}qv.
\]

P1 triangles give

\[
A_{ij}=\int_\Omega k\nabla\phi_i\cdot\nabla\phi_j+c\phi_i\phi_j,
\qquad M_{ij}=\int_\Omega\phi_i\phi_j,
\qquad F_i=\int_\Omega f\phi_i+\int_{\Gamma_N}q\phi_i.
\]

Eliminate prescribed DOFs \(B\): \(A_{II}U_I=F_I-A_{IB}g_B\).
The free set \(I\) includes unconstrained Neumann boundary nodes. The reduced
operator exposes residual, energy and lifting methods for later physics losses.
No analytical reference is passed to assembly or the solve.

## Manufactured cases

All three cases use \(k=1+x+y\) on the unit square.

| Case | Exact solution | Reaction | Boundaries |
|---|---|---|---|
| `variable_sine` | \(1+x+2y+\sin(\pi x)\sin(\pi y)\) | \(1+x\) | Dirichlet everywhere |
| `affine_diffusion` | \(1+2x-3y\) | 2 | Dirichlet everywhere |
| `mixed_polynomial` | \(1+x^2+y^2\) | 1 | Neumann at x=1; Dirichlet elsewhere |

Let \(s=\sin(\pi x)\sin(\pi y)\). For `variable_sine`,

\[
f=2\pi^2(1+x+y)s-3
-\pi\cos(\pi x)\sin(\pi y)-\pi\sin(\pi x)\cos(\pi y)
+(1+x)(1+x+2y+s).
\]

For `affine_diffusion`, \(f=3+4x-6y\). The exact solution belongs to the
finite-element space, providing a patch test with variable diffusion, reaction,
nonzero forcing and nonzero boundary values.

For `mixed_polynomial`,
\(f=-4(1+x+y)-2x-2y+1+x^2+y^2\), and
\(q=(1+x+y)(2x,2y)\cdot n\). On x=1 this is \(2(2+y)\).

## Interfaces and implementation

- `physics/base.py`: `EllipticProblem`, `AnalyticalReference`, finite scalar sampling.
- `physics/boundary.py`: independent Dirichlet values, edge selector and Neumann data.
- `physics/diffusion.py`: manufactured cases and analytically derived forcing.
- `mesh/nonuniform.py`: tensor meshes and graded axes \(x_i=(i/n)^p\).
- `fem/geometry.py`, `quadrature.py`: PDE-independent geometry and integration.
- `fem/boundary.py`: exterior edges, outward normals through edge orientation,
  and boundary partitioning. A Dirichlet selector acts on edge midpoints;
  its selected edges prescribe both endpoints, including shared corners.
- `fem/assembly.py`, `operators.py`, `elliptic.py`: full assembly, reduced system,
  and sparse CPU solve. Each connected component requires a Dirichlet edge.
- `experiments/elliptic.py`: evaluation, plots, CSV rates and run orchestration.
- `io/manifest.py`: shared provenance for Phase 1 and Phase 2A runs.

`Mesh2D.boundary` always means the **geometric exterior**, not the Dirichlet set.
Use `solution.reduced.boundary` and `solution.assembly.boundary.neumann_edges`
for the physical partition. `FieldResult` remains a reference-comparison artifact;
its required `exact` array is supplied only during manufactured evaluation.

## Quadrature and metrics

The triangle rule maps tensor Gauss-Legendre nodes to barycentric coordinates
with the Duffy Jacobian. Order m uses m² points and integrates total polynomial
degree up to 2m-2 exactly. Assembly requires order >=2 and defaults to order 4; error integration uses
order 5 independently. Neumann edge integration uses order 4 Gauss-Legendre.
Check integration-order sensitivity when introducing rapidly varying data.

Mass, stiffness and load are assembled separately. Positive diffusivity and
nonnegative reaction are checked at quadrature points; this does not prove
positivity everywhere for arbitrary user-supplied coefficient functions.

Metrics include absolute/relative L2 and H1 seminorm errors, maximum nodal and
Dirichlet boundary errors, and absolute/relative algebraic residual. Relative
errors use a 1e-24 floor for squared reference norms; the residual denominator
uses a 1e-12 floor. Use absolute metrics for zero or very small references.
Convergence rates use the maximum triangle diameter, including graded meshes.
Rates are omitted once errors fall below 1e-12, where roundoff can dominate.

Phase 1 assembly retains its original three-point rule. Its reported error
metrics now use higher-order independent integration, so historical error
tables are not numerically identical even though the FEM solutions are unchanged.

## Running

After activating the installed project environment:

```bash
python -m pytest -q
python scripts/mesh_convergence.py --sizes 8 16 32
python scripts/run_phase2.py --config configs/experiments/elliptic_local.json
python scripts/run_phase2.py --case mixed_polynomial --grading 1.3 --convergence 8 16 32
python scripts/run_phase2.py --case affine_diffusion --nx 9 --ny 8
```

On this workstation the existing environment is `.pignn-venv`; in Bash activate
with `source .pignn-venv/bin/activate`, or in Fish use `activate.fish`.

Defaults create unique directories under `outputs/phase2/`. An explicit
`--output` directory must be empty. Artifacts are `field.npz`, `metrics.json`,
`comparison.png`, `manifest.json`, and optional `convergence.csv`/`.png`.
The manifest records resolved settings, mesh sizes, errors, dependency/Python
versions, CPU precision, elapsed time, Git revision and dirty status. A source
digest is also stored; unavailable Git metadata is explicitly null.

CLI values override matching JSON settings. Unless explicitly configured,
`ny` follows the resolved `nx`. Convergence studies always use square n×n grids,
independent of the single-run aspect ratio.

## Verification and remaining scope

Tests check analytical element mass/stiffness/load matrices, polynomial
quadrature integrals, agreement with the original Poisson assembler, an affine
patch on a perturbed mesh, mixed-boundary flux sign, boundary elimination,
graded-mesh convergence, error-quadrature sensitivity, reduced energy gradients,
malformed inputs, archive round-trips and CLI provenance.

The included mesh generators cover the unit square with nonuniform triangles
and a polygonal annulus with tagged inner/outer boundaries (see `annulus.md`).
Imported meshes must already be conforming, nonoverlapping and CCW, with correct
exterior markers and edges aligned to boundary-type changes. Topological checks
do not detect every possible geometric overlap. Gmsh import,
tensor diffusion, discontinuous-interface integration, pure Neumann/nullspace
handling and Robin conditions remain unimplemented. FEniCSx comparisons passed
for the tested fixtures; the square-domain PINN baseline is described in `pinn.md`.

The geometry ladder now includes the annulus and a rectangle containing a
cylinder (see [cylinder milestone](cylinder.md)), plus a
[symmetric airfoil exterior](airfoil.md). Independent FEM comparison passed for the tested fixtures; initial CPU PINN training
and evaluation are now available in `pinn.md`. CUDA/GNN dependencies
remain deferred.

