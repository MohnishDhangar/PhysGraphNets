# No-slip obstacle Stokes validation — 30 October 2026

Scope: serial CPU float64, stationary P2/P1 Stokes with physical symmetric-gradient
stress, exact no-slip on polygonal obstacles and explicitly prescribed outer
velocity or outlet traction. See [equations, cases and acceptance](obstacle_stokes.md).
This qualifies initial finite-domain fixtures, not transient wakes or arbitrary shapes.

| Executed check | Result |
| --- | --- |
| Full installed-environment suite, `UCX_TLS=self .fenicsx-env/bin/python -m pytest -q` | 345 passed, 1 expected missing-dependency-path skip; 130.90 s |
| Obstacle tests in base `.pignn-venv` | 17 passed, 3 optional FEniCSx skips; 6.17 s |
| Three obstacle cases x levels 4/8/16, plus two annular radii x levels 8/16 | All 13 passed, with independent FEniCSx comparisons on every solve |
| Original Poisson convergence, n=8/16/32 | Relative L2 0.0422122 → 0.0107514 → 0.00270066 |
| Offline wheel build; isolated imports and no-slip channel solves from the wheel in both environments | Passed |

The qualification used Python 3.12.13, DOLFINx 0.10.0 and PETSc 3.25.6/MUMPS.
Base-environment checks used Python 3.14.7. All solves used nu=1, U=0.01 and
straight triangle geometry. Finest meshes have 4352 pressure vertices and 8192
triangles; velocity retains its additional P2 edge DOFs.

## Finest-mesh loads

Force is on the obstacle. Resistance divides force by nu U; it is not the
conventional drag coefficient based on inertial scaling. The final-pair drag
change and agreement of the two force estimators are diagnostics, not certified
continuum error bounds.

| Case | Reaction drag / (nu U) | Drag change, level 8→16 | Surface/reaction force disagreement | Absolute lift/drag |
| --- | ---: | ---: | ---: | ---: |
| Annular cylinder, R=2 | 13.3527402 | 0.02736% | 0.04014% | 1.20e-17 |
| Channel cylinder | 48.4400227 | 0.21225% | 0.16221% | 3.38e-6 |
| Symmetric airfoil, angular bias 0.2 | 8.8953709 | 0.68777% | 0.39667% | 0.0011204 |

The unchanged finest-mesh limits are 3% drag change, 5% force-estimator
disagreement and 2% absolute lift/drag. No-slip speed is exactly zero at all
obstacle velocity DOFs. Maximum net flux magnitude across all 13 solves is
2.64e-16; maximum algebraic force imbalance is 2.85e-13. Reconstructed total
surface-stress imbalance remains finite: relative to reaction-force magnitude,
it is 0.157%, 0.230% and 0.743% for these three finest solutions respectively.

## Analytical and external checks

For the annular cylinder at R=2, the finest reaction-drag error is 0.002313%
against the exact confined-cylinder solution. Velocity L2 error is 1.87397e-6
and pressure L2 error is 7.08049e-5, with final refinement orders 2.534 and 2.002.
These errors include polygonal geometry and boundary approximation; they are
not errors solely from discretizing the ideal circular domain.

At R=1.5 and 3, the finest normalized reaction drags are 18.3089572 and
9.5032584. Their analytical relative drag errors are 0.001459% and 0.004897%.
Each radius is compared with its own exact solution. This verifies confinement
dependence, not an infinite-domain drag limit. Channel outlet-length and airfoil
outer-radius sensitivity remain open.

Independent UFL/PETSc assembly on identical meshes gave these maximum absolute
differences over all 13 comparisons:

| Quantity | Maximum difference |
| --- | ---: |
| P2 velocity coefficients | 1.80e-14 |
| P1 pressure coefficients | 3.37e-12 |
| Per-tag integrated flux | 2.65e-16 |
| Per-tag surface-load components | 3.03e-13 |
| Obstacle reaction-force components | 7.37e-14 |

All meet the declared 1e-9 tolerance. Agreement between implementations on one
mesh does not establish its continuum accuracy; analytical and refinement checks
provide the separate evidence above. The channel and airfoil have no analytical
field/force ground truth in this increment.

## Airfoil failure and correction

The initial uniform-ray run failed the declared 5% force-estimator limit:
finest-mesh disagreement was 5.4489%, despite only 0.4814% change in reaction drag.
Its independently assembled FEniCSx comparison passed, pointing to a resolution
problem rather than disagreement between solvers. The largest surface edge was
0.07596 chord, leaving uneven body resolution.

The revised mesh clusters rays toward both profile endpoints with angular bias
0.2. At the same 8192 triangles, the largest body edge drops to 0.01668 chord.
The minimum triangle angle is 1.729 degrees and minimum scaled quality is 0.05150;
all orientation/topology checks pass. The final force-estimator gap is 0.3967%.
The physical problem, profile polynomial and acceptance limits are unchanged.
This is a tested mesh correction, not a claim of uniformly high mesh quality
or universal adequacy of this bias for other profiles and flow conditions.

## Reproducibility and next gate

Run:

```bash
UCX_TLS=self .fenicsx-env/bin/python scripts/run_obstacle_stokes.py \
  --fenicsx --output outputs/my_obstacle_qualification
```

The default study now uses the corrected airfoil mesh. The original mesh API and
scalar fixtures retain their uniform-ray defaults.

Generated evidence is retained in `outputs/obstacle_stokes_20261030/`:

- `qualification/`: initial failed study; no overwritten or discarded failure.
- `airfoil_bias_study/`: NumPy/SciPy diagnostic refinement with bias 0.2.
- `final_qualification/`: passing manifest and 13 tagged field/reaction archives.
- `fem_regression/`, `dist/` and `checks_manifest.json`: convergence, package and
  verification evidence.

Final qualification source SHA-256:
`f26f786174257f7de1c8278e22533885ed4d2534ad7ba8d810fe35fed82a32e0`.
Git revision: `1b1e773a417d9b520482e2283ef7057d985a30ef`, with local changes.
The manifest records full source-file coverage, parameters, checks and errors.
Its 197.07 s duration overlapped other checks and is not a performance benchmark.

Next: manufactured transient Navier-Stokes, time/space refinement and then
precisely specified obstacle wake validation. Expanded geometry/domain studies,
arbitrary-mesh import, triangle-based particles, fluid datasets and learned
rollout qualification remain open. GPU migration and final viewer work remain
deferred.
