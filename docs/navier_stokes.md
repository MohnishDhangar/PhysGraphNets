# Transient incompressible Navier–Stokes reference

This deliverable adds a CPU velocity–pressure solver for time-dependent flow on
fixed triangular meshes. The initial qualification covers manufactured square
problems with full velocity boundaries or known physical traction, plus unforced
closed-wall decay. It extends the [stationary obstacle reference](obstacle_stokes.md).
[Executed validation](validation_navier_stokes_20261030.md) records the evidence.
Transient obstacle wakes, arbitrary imported shapes and fluid GNN training remain
later gates. No dependency or hardware change is required.

## Mathematical contract

For constant positive kinematic viscosity nu, with density divided out:

    u_t + (u.grad)u - div(sigma(u,p)) = f,    div(u) = 0,
    sigma(u,p) = nu (grad(u) + grad(u)^T) - p I,
    u = g(x,t) on Gamma_D,    sigma n = traction(x,n,t) on Gamma_N,
    u(x,0) = u0(x).

The normal points outward from the fluid. Boundary tags form an exhaustive,
disjoint partition, with a nonempty full-vector Dirichlet boundary. Pressure and
integrated 2D loads are per unit density and out-of-plane depth. The physical
equations and stress convention follow the
[FEniCSx formulation](https://jsdokken.com/dolfinx-tutorial/chapter2/navierstokes.html).
The solver here uses a coupled implicit system; it does not use that tutorial's
pressure-correction splitting scheme.

With P2 velocity, P1 pressure and tests v vanishing on Gamma_D:

    (D_t u, v) + ((u.grad)u, v) + 2 nu (eps(u), eps(v))
        - (p, div(v)) - (q, div(u)) = (f,v) + <traction,v>_Gamma_N.

All nonlinear terms use the unknown velocity at the new time. Physical traction
is not replaced by the Laplacian conormal. Advection is in advective form, with
no skew-symmetric modification or artificial stabilization. Velocity is weakly
divergence-free; pointwise divergence is measured separately.

Full velocity data must have compatible net boundary flux at every time. The
solver then imposes zero mean pressure through a Lagrange multiplier. Prescribed
traction determines absolute pressure; mixed-boundary solves have no pressure
pin, pressure penalty or mean subtraction.

## Time stepping and nonlinear solution

The fixed-step options are:

    BE:   D_t u[n] = (u[n] - u[n-1]) / dt,
    BDF2: D_t u[n] = (3u[n] - 4u[n-1] + u[n-2]) / (2dt).

BDF2 starts with one BE step. Numerical velocities supply all subsequent history;
analytical interior fields are not injected into the time stepping. BDF methods
are standard implicit time integrators; see the
[PETSc BDF documentation](https://petsc.org/main/manualpages/TS/TSBDF/).
The NumPy/SciPy implementation here owns its time loop and does not invoke PETSc
TS. Time-order qualification is measured at the final time; the first pressure
sample still comes from the first-order startup step.

Newton's method includes both derivatives of advection:

    D[(u.grad)u] delta_u = (u.grad)delta_u + (delta_u.grad)u.

An independently checked Jacobian, sparse direct solve and residual-decreasing
backtracking determine each update. Boundary values are imposed at the new time
and Newton corrections vanish at prescribed velocity DOFs. The default relative
residual tolerance is 1e-12, with a maximum of 20 Newton updates. A failed solve
raises with step/time information; no unconverged state is accepted. The solver
records iterations, damping, momentum balance, flux and pressure policy per step.
There is no adaptive timestep controller or general high-Reynolds stability claim.

## Initial state and data contract

Simply sampling a continuous divergence-free field at P2 nodes need not produce
a weakly divergence-free discrete field. Initialization solves the constrained
L2 projection, with the prescribed velocity values at the initial time:

    (u_h0,v) - (lambda,div(v)) = (u0,v),    (q,div(u_h0)) = 0.

Initial data must match boundary data at prescribed nodes. The projection's
multiplier is not physical pressure and is discarded before time stepping.
`FlowTrajectory` holds the numerical histories. Its NPZ export stores:

- `times`, `velocity`: all N+1 states, including projected initial velocity.
- `pressure_times`, `pressure`: only the N solved times after initialization.
- P1 mesh and boundary tags, P2 velocity coordinates/connectivity, per-step
  momentum reactions and JSON metadata in pickle-free NPZ archives.

The in-memory `frame(step)` accepts steps 1 through N and exposes fields for
existing error and boundary-integral evaluation. It rejects step zero because
physical pressure has not been solved there. These archives are numerical
reference trajectories; a learning dataset contract and restart/checkpoint format
are separate pending tasks.

## Manufactured fields and quadrature

Let a(t) be exp(t) or 1+t. Cases use nu=0.1. The forcing is derived from the
continuous equation, including a'(t) and nonzero nonlinear advection.

| Case | Base velocity U; u=a(t)U | Base pressure P; p=a(t)P+C |
| --- | --- | --- |
| `accelerating_shear` | (1+y(1-y), 1/2) | x+y-1 |
| `vortex` | (psi_y, -psi_x), psi=32x²(1-x)²y²(1-y)² | x²+y²-2/3 |
| `translated_vortex` | vortex velocity + (1,-1/2) | x²+y²-2/3 |

The exact traction is evaluated from physical stress and the actual facet normal.
C=0 for full velocity and C=2.5 for mixed study cases. Tests check spatial and
temporal derivatives by finite differences, nonzero convection, all four traction
normals, pressure shifts and mesh/DOF permutation.

The accelerating-shear field is exactly representable by P2/P1, so its exponential
profile measures temporal error without an independent spatial approximation
floor. Its linear profile is a time-dependent exact patch. The two vortex fields
with linear amplitude test spatial refinement, including nonzero boundary lifting.
Their exact velocities have zero BE/BDF2 temporal consistency defect; the spatial
study still records its finite timestep explicitly.

The default Duffy quadrature order is 9, exact through total degree 16. The vortex
advection forcing times a P2 test can reach degree 15. This exceeds the degree
needed by the discrete convection matrix itself, so using only the latter's
degree to choose quadrature would underintegrate the manufactured load.

## Momentum and energy checks

Dirichlet reactions are the unreduced discrete momentum residual, including both
time derivative and advection. The total balance is

    body force + prescribed traction + Dirichlet reaction
        - integral(D_t u + (u.grad)u) = 0.

Surface-stress balance subtracts the same inertial contribution. It is a
reconstruction diagnostic, distinct from algebraic balance; treating an unsteady
solution as a static force balance would give the wrong interpretation.

For unforced zero-wall runs, let ||u||_M²=u^T M u and E=||u||_M²/2. BE checks

    (E[n]-E[n-1])/dt + ||u[n]-u[n-1]||_M²/(2dt)
        + viscous dissipation + advective power = 0.

After startup, BDF2 uses

    G[n] = (||u[n]||_M² + ||2u[n]-u[n-1]||_M²)/4,
    (G[n]-G[n-1])/dt + ||u[n]-2u[n-1]+u[n-2]||_M²/(4dt)
        + viscous dissipation + advective power = 0.

Advective power is retained because weak incompressibility does not make it
identically zero. The tested closed-wall cases also require decreasing physical
kinetic energy. This is evidence for those cases, not a proof of unconditional
energy stability, backflow stability or turbulent-flow accuracy. The energy
diagnostic rejects nonzero assembled loads, including loads whose net force
vanishes.

## Independent verification and reproduction

`verification/navier_stokes.py` constructs separate UFL expressions for exact
fields, time derivatives, forcing, traction and the nonlinear residual. It
performs its own initial projection and every timestep with PETSc/MUMPS and
SNES, using UFL automatic differentiation for the Jacobian. This uses the
[installed DOLFINx nonlinear-problem interface](https://docs.fenicsproject.org/dolfinx/v0.10.0.post5/python/generated/dolfinx.fem.petsc.html).
Only the mesh, physical specification and time grid are shared with the CPU
solver; numerical velocity history is independently evolved.

Comparison includes initial projected velocity, every solved velocity/pressure
field, per-tag flux and stress loads, and total Dirichlet reactions. Full-boundary
external pressure uses a pin and subsequent mean removal, independently checking
our mean constraint. Mixed pressure is compared at its absolute level.

```bash
UCX_TLS=self .fenicsx-env/bin/python scripts/run_navier_stokes.py \
  --fenicsx --output outputs/my_transient_qualification
```

Omit `--fenicsx` for the base NumPy/SciPy environment. The per-command UCX setting
allows serial MPI in the restricted environment. The output directory must be
empty. `--sizes` and `--time-steps` select successively doubled study sequences.
The default study comprises six exact patches, twelve temporal trajectories,
twelve spatial trajectories and two closed-wall decay trajectories. All 30
manufactured trajectories receive external comparisons when requested.

The manifest declares thresholds before execution: BE final-time velocity and
pressure orders >=0.85; BDF2 orders >=1.75; spatial velocity L2 >=2.5, H1,
pressure L2 and divergence L2 >=1.8. Patch errors <=1e-9, boundary error <=1e-12,
weak divergence/net flux <=1e-10, momentum balance <=1e-9, and independent
differences <=1e-9. Each run preserves statuses, errors, gates, mesh/time settings,
versions, source hash and revision. No renderer or neural prediction participates.

## Next gates

1. Qualify a precisely specified transient cylinder problem, with compatible
   startup, wall/inlet/outlet treatment, domain/mesh/timestep studies and force
   histories against an independent benchmark. Generated-mesh audits do not
   establish arbitrary-mesh import or high-Reynolds adequacy.
2. Add triangle-based particle location/interpolation and explicit wall/outlet
   policies, using the numerical velocity history.
3. Define transient vector/pressure datasets, checkpoint/restart policy, fluid
   models and held-out rollout tests. The scalar square GNN is not a fluid model.
4. Profile the verified solver and training workload before deciding on cloud
   GPUs. Keep the viewer and final video work for the final integration phase.
