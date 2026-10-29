# Boundary-aware Stokes reference

This is the first velocity/pressure FEM milestone, preceding transient obstacle
flow and fluid learning. Viewer, rendering and final video work are deferred by
user direction until the primary numerical and learned-model deliverables qualify.

## Formulation and scope

For constant positive dimensionless kinematic viscosity nu, solve

    -nu Delta(u) + grad(p) = f,    div(u) = 0 in Omega,
    u = g on the entire exterior boundary,    integral_Omega p = 0.

Here p is pressure divided by constant density. Full velocity Dirichlet data
must satisfy zero net boundary flux. There is no pressure penalty, artificial
compressibility or learned replacement of reference values.

Use continuous vector P2 velocity and scalar P1 pressure (Taylor-Hood) on straight
triangles. For velocity tests v vanishing on the boundary and pressure tests q:

    nu integral grad(u):grad(v) - integral p div(v) = integral f.v,
    -integral q div(u) + lambda integral q = 0,
    integral p = 0.

The scalar multiplier fixes the pressure nullspace. Compatible boundary data
give lambda approximately zero. The discrete prescribed net flux is checked
before solving so the multiplier cannot silently absorb incompatible data.
Nonzero velocity values are eliminated with their full contributions to both
momentum and continuity right-hand sides.

The implementation uses standard physical pressure sign, explicitly unlike the
sign convention used in portions of the
[DOLFINx Taylor-Hood demo](https://docs.fenicsproject.org/dolfinx/main/python/demos/demo_stokes.html).
Quadratic velocity with linear pressure follows that established mixed-element
family; the code here independently assembles it in NumPy/SciPy.

The original milestone qualifies unit-square cases with **all boundary velocities
prescribed**. The channel case prescribes its exact profile at both ends; it
does not qualify a natural-traction/open outlet. The subsequent
[mixed-boundary milestone](stokes_mixed.md) adds physical traction, absolute
pressure and independently checked boundary fluxes/loads. The solver checks positive
triangle orientation, exterior boundary masks, edge incidence, connectedness and
provided edge-tag coverage. These checks do not replace a future geometric mesh
import validator for intersections, overlap, slivers and near-wall resolution.

## Manufactured cases

All pressure formulas below have mean zero on the unit square.

| Case | Velocity | Pressure / forcing | Purpose |
| --- | --- | --- | --- |
| affine | (1+x, -1/2-y) | p=x+y-1; f=(1,1) | Exact affine patch, nonzero boundary and pressure sign |
| channel | (y(1-y), 0) | p=nu(1-2x); f=0 | Exact parabolic profile, wall values and viscous pressure balance |
| vortex | (psi_y, -psi_x), psi=x²(1-x)²y²(1-y)² | p=x²+y²-2/3; f=-nu Delta(u)+(2x,2y) | No-slip boundary, nontrivial velocity/pressure refinement |
| translated_vortex | vortex+(1,-1/2) | Same p and f as vortex | Nonzero Dirichlet lifting with the same interior discretization error |

Polynomial derivatives are evaluated in `physics/stokes.py`. Tests independently
check derivatives/forcing by finite differences. Assembly is in `fem/stokes.py`;
error integration in `evaluation/stokes.py`; experiments and CLI orchestration
remain separate. Velocity coefficients live at vertices and edge midpoints;
pressure coefficients live only at vertices. They must not be treated as one
shared scalar field or stripped down to P1 velocity for validation.

The default assembly rule has order 6 (degree 10 exactness); the error rule has
order 9 (degree 16 exactness). These integrate the polynomial fixture terms and
squared errors without underintegration. Errors are physical L2 velocity,
velocity H1 seminorm, pressure L2 and pointwise-divergence L2. Taylor-Hood enforces
incompressibility weakly: a small algebraic continuity residual does not imply
pointwise zero divergence. Both metrics are reported.

## Independent comparison and experiment

`verification/stokes.py` uses independent UFL differentiation and assembly on the
identical triangle mesh, and PETSc/MUMPS for the solve. It pins one pressure DOF,
then subtracts its integrated pressure mean. This differs from our mean-constraint
solve and checks gauge independence. Coordinate bijections align both P2 velocity
and P1 pressure DOFs; comparison tolerance is 1e-9. The working adapter targets
the installed DOLFINx 0.10 environment, serial real float64 PETSc with MUMPS.

```bash
UCX_TLS=self .fenicsx-env/bin/python scripts/run_stokes.py \
  --sizes 4 8 16 --fenicsx --output outputs/my_stokes_qualification
```

`UCX_TLS=self` enables serial MPI in the restricted local execution environment;
it is a per-command setting, not a change to the user's environment. Omitting
`--fenicsx` runs the NumPy/SciPy study without the optional external stack.

The runner refuses nonempty output directories and writes a manifest for running,
failed or completed studies. It records Python/dependency versions, code revision
and source hash, mesh sizes, viscosity, precision, errors, timings and gates.
NPZ fields retain P2 coordinates/connectivity, P1 mesh, case and configuration.
These are numerical artifacts; no rendering is produced.

Acceptance is declared in the runner: patch errors <=1e-10, residual <=1e-9,
boundary error <=1e-12, pressure mean and weak-divergence residual <=1e-10.
On the finest refinement pair, required orders are >=2.5 for velocity L2 and
>=1.8 for velocity H1, pressure L2 and divergence L2. These checks establish the
manufactured square milestone, not universal mesh stability or obstacle accuracy.

## Next numerical deliverables

1. Complete: [tagged velocity/traction conditions](stokes_mixed.md), channel
   boundary verification and independent flux/load integration.
2. Complete for initial fixtures: [no-slip cylinder/airfoil Stokes](obstacle_stokes.md),
   generated-mesh audits, independent loads and boundary refinement. Arbitrary
   import and broader obstacle/domain coverage remain open.
3. Complete for manufactured cases: [transient Navier-Stokes](navier_stokes.md),
   timestep/mesh studies and independent trajectories. Next: a precisely specified
   transient cylinder benchmark.
4. Boundary-aware particle sampling and transport verification.
5. Transient fluid dataset contracts, CPU models and held-out rollout qualification.
6. Interactive viewer and final video export after the primary pipeline qualifies.
