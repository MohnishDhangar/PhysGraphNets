# Mixed Stokes validation — 29 October 2026

Scope: serial CPU float64 Taylor-Hood, physical symmetric-gradient stress,
prescribed velocity on three square sides and exact traction on the fourth.
Both left and right traction orientations are qualified. Pressure is determined
by traction, without pinning or mean subtraction. See [formulation and primary
sources](stokes_mixed.md).

| Executed check | Result |
| --- | --- |
| Full installed-environment suite, `UCX_TLS=self .fenicsx-env/bin/python -m pytest -q` | 325 passed, 1 expected missing-dependency-path skip; 118.17 s |
| New focused mixed-boundary tests | 25 passed |
| Both Stokes test modules in base `.pignn-venv` | 40 passed, 4 optional FEniCSx skips; 3.63 s |
| Four manufactured cases × two traction sides × n=4/8/16, independent FEniCSx comparison on every case | All 24 passed |
| Original Poisson convergence at n=8/16/32 | Completed; relative L2 0.0422122 → 0.0107514 → 0.00270066 |
| Offline wheel build and isolated mixed-channel pressure/flux solve from the wheel | Passed |
| Documentation links and whitespace | Passed |

The study used nu=0.3 and a pressure offset of 2.5. FEniCSx comparisons used the
installed DOLFINx 0.10.0 and PETSc 3.25.6/MUMPS. Maximum absolute differences
across all comparisons were:

| Compared quantity | Maximum difference |
| --- | --- |
| P2 velocity coefficients | 6.95e-14 |
| P1 pressure coefficients | 6.64e-13 |
| Per-tag integrated boundary flux | 5.11e-15 |
| Per-tag integrated boundary load components | 4.85e-13 |

All are below the predeclared 1e-9 external-comparison threshold. The maximum
net boundary flux magnitude was 3.33e-15; the maximum algebraic body-force,
applied-traction and Dirichlet-reaction imbalance was 5.64e-13.

For the manufactured vortex with right traction at n=16:

| Quantity | Absolute error | Observed order, n=8 to 16 |
| --- | --- | --- |
| Velocity L2 | 5.59117e-6 | 3.112 |
| Velocity H1 seminorm | 6.53628e-4 | 1.972 |
| Pressure L2 | 4.12016e-4 | 2.004 |
| Divergence L2 | 4.66157e-4 | 1.909 |
| Boundary traction L2 | 1.97342e-3 | 1.942 |

Maximum per-tag manufactured load error was 6.71e-4. The independently
reconstructed stress-force imbalance was 4.31e-5. Those nonzero discretization
errors are distinct from the algebraic balances above and must not be presented
as roundoff-level physical force accuracy. The field remains weakly rather than
pointwise divergence-free.

The affine/channel cases recover their representable polynomial solutions and
absolute pressure levels to floating-point accuracy. Unit tests independently
cover all four traction-side normals, pressure shifts with invariant velocity,
nonfinite/missing/overlapping data, edge/node permutation, reversed tag orientation,
zero-traction extension flow and normals on a polygonal hole. The hole fixture
is a manufactured topology check, not a no-slip obstacle-flow benchmark.

Final artifacts are in `outputs/stokes_mixed_20261029/final_qualification/`,
including the manifest and 24 tagged NPZ fields. The manifest records Python,
dependencies, source/revision identity, parameters, mesh sizes, checks and errors.
`checks_manifest.json`, `fem_regression/` and `dist/` hold regression/build evidence.
The final qualification took 25.38 s while other checks ran concurrently; this is
an experiment duration, not a controlled performance benchmark. The initial
passing study is retained in `qualification/`; the final run includes explicit
physical-traction/conormal labels in the boundary diagnostics.

The next numerical gate is a stationary no-slip obstacle reference with mesh,
domain and boundary refinement and independent load checks. Transient outlet
stability, cylinder wakes, moving boundaries and fluid-model training remain open.
