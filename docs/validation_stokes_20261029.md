# Stokes milestone validation — 29 October 2026

Scope: serial CPU float64, unit-square P2/P1 Taylor-Hood, full prescribed velocity
on every exterior boundary, mean-zero pressure. See [formulation](stokes.md).
This is not qualification of open outlets, obstacle aerodynamics or transient flow.

| Executed check | Result |
| --- | --- |
| Full installed-environment suite, `UCX_TLS=self .fenicsx-env/bin/python -m pytest -q` | 300 passed, 1 expected missing-dependency-path skip; 92.07 s |
| Focused Stokes suite in installed environment | 19 passed |
| Focused Stokes suite in base `.pignn-venv` | 17 passed, 2 optional FEniCSx skips |
| Four manufactured cases at n=4/8/16, each with independent FEniCSx comparison | All 12 passed |
| Existing Poisson mesh convergence, n=8/16/32 | Completed; relative L2 0.0422122 → 0.0107514 → 0.00270066 |
| Offline wheel build and isolated import/channel solve from the wheel | Passed |
| Documentation links and whitespace | Passed |

Independent comparison used DOLFINx 0.10.0 and PETSc 3.25.6/MUMPS. Across the 12
comparisons, maximum coefficient differences were 8.23e-14 for velocity and
1.13e-11 for pressure, below the predeclared 1e-9 threshold. No neural values enter
these checks. The independent solve pins pressure, then removes its integrated
mean; the local solve uses a mean constraint.

For the no-slip manufactured vortex at n=16 (512 triangles, 1,089 velocity nodes,
289 pressure nodes):

| Quantity | Absolute error | Observed order, n=8 to 16 |
| --- | --- | --- |
| Velocity L2 | 5.30146e-6 | 3.008 |
| Velocity H1 seminorm | 6.52579e-4 | 1.966 |
| Pressure L2 | 4.12448e-4 | 2.017 |
| Divergence L2 | 4.74129e-4 | 1.940 |

The translated vortex retains these errors while exercising nonzero boundary
data. The polynomial affine/channel patch cases recover their exact representable
solutions to floating-point accuracy. For the n=16 vortex, weak-divergence residual
was 6.77e-17 and pressure mean 6.03e-17. The nonzero divergence L2 above is expected:
Taylor-Hood is weakly, not pointwise, divergence-free.

Artifacts are under `outputs/stokes_20261029/`: `qualification_pivoted/manifest.json`
and its NPZ fields, `fem_regression/`, `checks_manifest.json`, and `dist/`.
Manifests include source identity, runtime environment, mesh sizes and gates.
The external qualification took 27.47 s including external compilation/setup;
this is not an isolated solver-performance benchmark.

The first external attempt remains in `qualification/manifest.json` with failed
status: PETSc's default LU encountered a zero pivot in the indefinite mixed system.
The adapter now explicitly requires the installed MUMPS pivoted factorization;
no numerical stabilization or tolerance relaxation was introduced. A constant-zero
UFL boundary expression also needed an explicit communicator. Both issues were
resolved before the successful full regression run.

Next gate: mixed velocity/traction boundaries, pressure policy and integrated
flux/load checks, followed by fixed-obstacle reference qualification. Viewer and
final-video implementation remain deferred.
