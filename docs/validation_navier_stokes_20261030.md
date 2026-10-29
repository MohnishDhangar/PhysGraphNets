# Transient Navier–Stokes validation — 30 October 2026

Scope: fixed triangular meshes, CPU float64 P2/P1, fully implicit nonlinear
advection, physical symmetric-gradient stress, backward Euler and fixed-step
BDF2 with one BE startup step. The cases are manufactured square flows with
prescribed velocity or exact traction, plus unforced closed-wall decay.
See [formulation and primary sources](navier_stokes.md).

| Executed check | Result |
| --- | --- |
| Full installed-environment suite, `UCX_TLS=self .fenicsx-env/bin/python -m pytest -q` | 385 passed, 1 expected missing-dependency-path skip; 132.38 s |
| Transient module in base `.pignn-venv` | 38 passed, 2 optional FEniCSx skips; 6.13 s |
| Six patches, twelve temporal trajectories and twelve spatial trajectories | All 30 passed, with independent FEniCSx comparisons at every solved step |
| Independently compared trajectory steps | 296, plus all 30 initial velocity projections |
| Unforced closed-wall BE and BDF2 decay | Both passed energy identities and monotonic physical-energy checks |
| Finest spatial-case timestep sensitivity | Passed; maximum relative change among the four error norms was 0.0212% |
| Original Poisson convergence, n=8/16/32 | Relative L2 0.0422122 → 0.0107514 → 0.00270066 |
| Offline wheel, isolated imports and transient mixed-boundary solves in both environments | Passed |
| Pickle-free readback of all 32 qualification archives | Passed |

The working environment used Python 3.12.13, DOLFINx 0.10.0 and PETSc
3.25.6/MUMPS; the base environment used Python 3.14.7. No dependencies were added.

## Temporal and spatial accuracy

The temporal experiment uses exactly representable quadratic velocity and linear
pressure with exponential amplitude, including nonzero advection. It runs to
T=0.4 with dt=0.05, 0.025 and 0.0125 on n=4. Rates below use the finest pair
and final-time errors; they do not assert second-order pressure at the BE startup.

| Method / boundary | Velocity L2 order | Pressure L2 order |
| --- | ---: | ---: |
| BE / full velocity | 0.991 | 0.994 |
| BE / right traction | 0.987 | 0.994 |
| BDF2 / full velocity | 2.008 | 1.987 |
| BDF2 / right traction | 2.036 | 1.986 |

Predeclared minima are 0.85 for BE and 1.75 for BDF2. Finest BDF2 final velocity
L2 errors are 2.87456e-7 and 2.40363e-6 for full velocity and right traction.
Pressure errors are 2.87387e-5 and 5.27450e-5 respectively.

Spatial experiments use n=4,8,16, dt=0.025 and T=0.1, with a linear manufactured
amplitude having zero BE/BDF2 consistency defect. All numerical history is evolved
from the projected initial state. The final-pair spatial orders are:

| Case / boundary | Velocity L2 | Velocity H1 seminorm | Pressure L2 | Divergence L2 |
| --- | ---: | ---: | ---: | ---: |
| Vortex / full velocity | 3.014 | 1.971 | 2.320 | 1.919 |
| Vortex / right traction | 3.002 | 1.967 | 2.398 | 1.906 |
| Translated vortex / full velocity | 3.026 | 1.973 | 2.389 | 1.920 |
| Translated vortex / right traction | 3.030 | 1.969 | 2.445 | 1.907 |

All meet the declared minima of 2.5, 1.8, 1.8 and 1.8. The finest translated
mixed case has velocity L2 error 1.89515e-4, velocity H1 error 2.29653e-2,
pressure L2 error 4.98101e-4 and divergence L2 1.63509e-2. The nonzero pointwise
divergence must not be confused with the algebraic weak-divergence residual.

For that same case, halving dt to 0.0125 changed these errors by 0.00889%,
0.000521%, 0.02120% and 0.000793% respectively. The separate sensitivity run
declared a 5% limit before execution and passed it. This supports the spatial
interpretation for this tested case; it is not an arbitrary-flow timestep bound.

## Independent trajectories, boundaries and balances

The UFL/SNES comparison performs its own constrained initial projection, uses
independently differentiated manufactured forcing and nonlinear Jacobian, and
evolves its own history. It compares all solved fields, fluxes, surface loads and
total Dirichlet reactions. Maximum differences across the 30 trajectories were:

| Quantity | Maximum absolute difference |
| --- | ---: |
| P2 velocity coefficients, including initial projections | 6.67e-12 |
| P1 pressure coefficients | 5.19e-11 |
| Per-tag integrated flux | 8.44e-15 |
| Per-tag surface-load components | 1.70e-11 |
| Total Dirichlet-reaction components | 2.59e-11 |

All are below the predeclared 1e-9 threshold. Full-velocity pressure is aligned
by its integrated mean; mixed pressure is compared without subtracting a mean.
The analytical references remain separate from the numerical solution.

Across the manufactured trajectories, maximum nonlinear relative residual was
6.45e-13, weak-divergence residual 2.30e-16, integrated net flux magnitude
1.11e-15, and algebraic momentum imbalance 2.65e-11. Prescribed velocity error
was zero. At most three Newton updates were needed per step. These counts are
observations on the qualified fixtures, not a general nonlinear-convergence claim.

The exact linear-time patches recover velocity, gradients and pressure to
floating-point accuracy. Unit tests additionally cover all four traction-side
normals, boundary mappings and corner conflicts, changing-time flux incompatibility,
initial/boundary mismatch, pressure shifts, the advection Jacobian by finite
differences, mesh/DOF permutations, invalid inputs, and refusal of unconverged
Newton steps. A steady channel reproduces the existing Stokes reference.

## Closed-wall energy

Both decay experiments use n=6, dt=0.025 and T=0.5, starting from a projected
vortex with zero forcing and homogeneous velocity boundaries. Kinetic energy
starts at 0.03095594 and decreases at every step to:

- BE: 0.0002160653; maximum energy-identity residual 1.23e-14.
- BDF2: 0.0001561416; maximum energy-identity residual 1.87e-14.

The identities include BE/BDF2 numerical dissipation and the measured advective
power. They do not assume pointwise divergence-free velocity or prove stability
for arbitrary open boundaries. The diagnostic also rejects nonzero assembled
loads whose integrated force happens to cancel.

## Artifacts and remaining scope

The final manifest and 32 trajectory archives are in
`outputs/transient_navier_stokes_20261030/final_qualification/`. Each velocity
history includes its initial state. Physical pressure and reaction histories
begin at the first solved time; no projection multiplier is exported as pressure.
The archives preserve P2 connectivity and complete boundary tags.

`spatial_timestep_sensitivity/`, `fem_regression/`, `dist/` and
`checks_manifest.json` contain the additional checks. `adapter_smoke/` and
`qualification/` retain development runs; the final run includes the stricter
energy-load diagnostic and its regression tests.

Final source SHA-256:
`e62ebd68c7d7760586c06add2bf7a1904567f1182a938dbd7c555a01096257ac`.
Git revision: `1b1e773a417d9b520482e2283ef7057d985a30ef`, with local changes.
The 59.11 s study duration overlapped other work and is not a controlled speed
benchmark. Source identity was checked again after qualification.

This completes the manufactured transient solver gate. It does not qualify
cylinder wakes, arbitrary meshes, high-Reynolds turbulence, moving boundaries,
transient fluid learning or final rendering. Next is a precisely specified
transient cylinder benchmark with compatible startup, mesh/time/domain studies
and independent force histories, followed by triangle-based particles and
transient datasets. GPU migration and the viewer remain deferred.
