# Independent FEniCSx comparison

**The external comparison passed on 26 October 2026:** all 24 runs across
12 fixtures and two resolutions passed in `.fenicsx-env` (DOLFINx 0.10.0,
Python 3.12.13). The full suite passed 161 tests with one expected missing-
dependency-path skip. See [the validation record](validation_20261026.md).
A run in an environment without FEniCSx still writes
`status: blocked_missing_dependency` and exits 2.

## Scope and independence

`verification/fenicsx.py` targets real-double PETSc with DOLFINx 0.10/0.11 on one
MPI process. It imports the exact PhysGraph triangle coordinates/connectivity,
then independently constructs the P1 space, UFL stiffness/mass/load forms,
Dirichlet DOFs, outward facet normals, boundary fluxes, and PETSc LU solve.
Forcing is derived symbolically as -div(k grad(u_exact))+c*u_exact, rather than
calling PhysGraph's explicit manufactured forcing. DOLFINx's own quadrature
also computes continuum L2/H1 errors. The external solve never calls PhysGraph
assembly, quadrature, basis-gradient, boundary-elimination or solve functions.

Sharing the geometry is deliberate: differing triangulations would conflate
assembly/solver differences with mesh approximation error. The adapter checks a
bijective coordinate mapping because DOLFINx can reorder vertices and DOFs.
Facet tags are transferred by edge endpoints, never by assumed index order.
Dirichlet node sets must independently agree. Shared input tags mean this is
not an independent test of the mesh generator itself; existing geometry tests
cover that responsibility.

The default suite includes 12 fixtures at resolutions 4 and 8:

- Graded square: `variable_sine`, `affine_diffusion`, `mixed_polynomial`.
- Annulus: all Dirichlet, inner Neumann, outer Neumann.
- Rectangle/cylinder: all Dirichlet, cylinder Neumann, outer Neumann.
- Symmetric airfoil exterior: all Dirichlet, airfoil Neumann, far-field Neumann.

Geometry fixtures use their documented default parameters. Polynomial fields,
annulus log fields, nonzero Dirichlet values, and both flux orientations are
included. This adapter is a fixture comparison, not a general callable-PDE
translator or distributed benchmark.

## Acceptance and artifacts

For each fixture the runner reports maximum nodal solution difference,
mass-weighted relative solution difference, full stiffness/mass-matrix differences,
load differences, boundary errors, and analytical errors of both solutions.
The matrix metrics use Frobenius norms. The load metric divides by max(||F||,1),
so zero-load cases use an absolute measure. The default tolerance is 1e-8 for
all parity checks, including absolute nodal and boundary differences.

PhysGraph assembly uses order 7 (49 points per triangle); external forms use
quadrature degree 12. These are different integration rules. Error parity
for nonpolynomial data should also be checked under increased quadrature if
the default tolerance fails; do not loosen a threshold to hide a formulation
or boundary mismatch.

Saved NPZ files contain distinct `physgraph_fem`, `fenicsx_fem`, and `exact`
arrays on shared nodes/triangles. JSON manifests record cases, mesh sizes,
settings, versions, errors, source digest/revision and pass/fail/blocked status.
No neural field is used as an FEM reference. Optional integration tests check
both nodal/matrix parity and the separately integrated external L2/H1 errors;
without FEniCSx those tests are explicitly skipped.

## Isolated installation and execution

The [official FEniCS installation guide](https://fenicsproject.org/download/)
and [DOLFINx installation options](https://github.com/FEniCS/dolfinx#installation)
recommend binary environments including conda. The adapter follows the
[DOLFINx 0.10 Poisson API](https://docs.fenicsproject.org/dolfinx/v0.10.0.post3/python/demos/demo_poisson.html).
Use a separate environment so the lightweight CPU baseline remains available:

```bash
conda env create --prefix .fenicsx-env --file configs/environments/fenicsx.yml
conda run --prefix .fenicsx-env python -m pip install -e '.[dev]'
conda run --prefix .fenicsx-env python -m pytest -q tests/test_fenicsx_comparison.py
conda run --prefix .fenicsx-env python scripts/compare_fenicsx.py --resolutions 4 8
```

The supplied environment pins the targeted 0.10 API and real PETSc. It is now
installed and verified locally as `.fenicsx-env`. Do not replace it with the legacy `fenics` pip
package; that is a different project/API. The adapter rejects unsupported
DOLFINx versions, complex PETSc builds, and multiple MPI processes.

In a restricted shell, MPICH/UCX may abort during import because socket creation
is prohibited. For this adapter's **single-process** runs, `UCX_TLS=self` selects
UCX's within-process transport; it does not require communication between ranks.
See the [UCX transport documentation](https://openucx.readthedocs.io/en/master/faq.html).
This is an execution setting, not a change to numerical code:

```bash
XDG_CACHE_HOME=/tmp/physgraph-cache MPLCONFIGDIR=/tmp/physgraph-mpl \
OPENBLAS_NUM_THREADS=1 UCX_TLS=self \
  "$HOME/.local/bin/micromamba" run -p "$PWD/.fenicsx-env" python -m pytest -q
```

A focused check is:

```bash
python scripts/compare_fenicsx.py --cases affine_diffusion mixed_polynomial --resolutions 4 8 16
```

Output defaults to unique directories under `outputs/fenicsx/`; explicit
`--output` directories must be empty. Exit 0 means every comparison passed;
exit 2 indicates missing dependencies or failed parity. Other adapter/runtime
failures also exit nonzero and preserve a failure manifest when the runner has
started. The recorded successful run closes this gate for the tested fixtures
and resolutions; extensions require their own independent checks.

## Recorded execution status

The original attempt in the lightweight environment was blocked and remains
recorded at `outputs/phase2b_validation/fenicsx/manifest.json`. After installation,
the user executed the 12 integration tests and the 24-case/resolution comparison.
The successful manifest is
`outputs/fenicsx/20261026T181836Z_c2b6cdc5/manifest.json`. Maximum nodal difference
was 1.850742e-13, below the configured 1e-8 tolerance. Matrix, mass, load and
boundary checks also passed. [Full evidence and scope](validation_20261026.md).
