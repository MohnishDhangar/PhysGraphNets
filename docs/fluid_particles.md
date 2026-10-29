# Fluid flow and passive particles: first verified reference

User priority: fluid flow/particle motion first; heat/diffusion remains a future
viewer mode. This increment adds a CPU reference solver and time-indexed playback
artifacts. It does not train the scalar elliptic GNN to predict fluid motion.

## Mathematical formulation

On the periodic square [0,2*pi)^2, with density one and viscosity nu>0:

    u_t + (u.grad)u = -grad(p) + nu Delta(u),    div(u)=0.
    omega = dv/dx - du/dy
    omega_t + u.grad(omega) = nu Delta(omega)
    -Delta(psi) = omega,    u = (psi_y, -psi_x).

There is no body force. The solver supports zero mean velocity and zero mean
vorticity; it rejects incompatible mean vorticity. Pressure is eliminated by the
curl formulation and is not an exported field. Periodic boundaries identify
opposite sides; these are not no-slip walls or inflow/outflow boundaries.
For the velocity/pressure formulation and why boundary conditions matter, see
the [FEniCSx tutorial](https://github.com/jorgensd/dolfinx-tutorial/blob/main/chapter2/navierstokes.md)
and [DOLFINx incompressible-flow demo](https://docs.fenicsproject.org/dolfinx/v0.9.0/python/demos/demo_navier-stokes.html).
Those sources are background; this implementation is a NumPy Fourier reference,
not an execution of those finite-element demos.

`fluids/periodic.py` uses FFT differentiation, a strict 2/3 Fourier cutoff to
remove product aliasing, and classical explicit RK4. Initial data are projected
onto the retained modes. A conservative combined advection/diffusion restriction
is checked at all four stages; oversized time steps fail instead of continuing.
This is an initial resolved-flow reference, not a turbulent or arbitrary-domain
solver. Grid fields have axis order [y,x,component].

Manufactured cases in `physics/periodic_flow.py`:

- Taylor-Green: u=exp(-2*nu*t)*(sin(x)cos(y), -cos(x)sin(y)),
  omega=2*exp(-2*nu*t)*sin(x)sin(y). Pressure, if desired analytically, is
  exp(-4*nu*t)*(cos(2*x)+cos(2*y))/4 up to a constant.
- Decaying shear: u=exp(-nu*t)*(sin(y),0), omega=-exp(-nu*t)*cos(y).

Both solve the unforced equations exactly. Their vorticity-advection term
vanishes, so an additional two-mode streamfunction test independently checks a
**nonzero** nonlinear advection term against its analytic product-rule expression.
Tests also check signs, incompressibility, RK4 time convergence and energy decay.

## Tracers and animation

Massless, passive tracers satisfy dX/dt=u(X,t). They do not exert forces on the
fluid, collide, diffuse, settle, or model inertial particles. Periodic bilinear
interpolation samples the numerical velocity grid at each of the four coupled
RK stages. Positions are integrated without wrapping and also exported modulo
2*pi for display. For shear, the independent trajectory is

    X(t) = X(0) + sin(Y(0))*(1-exp(-nu*t))/nu,    Y(t)=Y(0).

Bilinear sampling limits spatial tracer accuracy to approximately second order,
even though fluid differentiation is spectral and time stepping is fourth order.
An initial position outside the box is interpreted periodically, not clipped.

`trajectory.npz` loads with `allow_pickle=False` and contains times, numerical
velocity/vorticity grids, wrapped/unwrapped particle positions, grid coordinates
and schema-1 JSON metadata. The manifest records Python/source identity,
viscosity, timestep, grid, initial-case identity, particle seed, analytical field
errors, mean kinetic energy/enstrophy and divergence. Analytical and numerical
fields are never replaced by neural predictions.

`particles.gif` shows numerical vorticity with a fixed color range and numerical
tracers. It is a scientific playback artifact, not the interactive viewer. Uniform
frame times are required for GIF export; playback fps need not match physical
seconds, and every frame shows physical time. Periodic seam crossings wrap on
screen; unwrapped coordinates support later continuous trajectory inspection.

```bash
.fenicsx-env/bin/python scripts/run_periodic_flow.py \
  --config configs/experiments/periodic_flow.json \
  --output outputs/my_periodic_flow
```

## Next fluid milestones

Viewer and final-video work are deferred by user direction until the primary
numerical and learned-model pipeline qualifies. The [Stokes milestone](stokes.md)
now provides a P2/P1 CPU reference with full prescribed boundary velocity and
independent FEniCSx comparisons. The [mixed-boundary extension](stokes_mixed.md)
adds physical traction, absolute pressure and boundary flux/load verification.
The [initial obstacle reference](obstacle_stokes.md) now qualifies stationary
no-slip cylinder/airfoil fixtures and generated-mesh audits. The
[transient reference](navier_stokes.md) now qualifies manufactured space/time
refinement, independent complete trajectories and closed-wall energy checks. The
[viewer plan](viewer_plan.md) is retained for the final integration phase.

1. Completed: manufactured transient Navier-Stokes with spatial and temporal refinement.
2. Expand fixed-obstacle parameter/domain studies; add arbitrary-mesh import
   with geometric intersection and complete boundary-tag validation.
3. Extend transient incompressible flow to obstacle meshes with stability,
   refinement, force/balance and external-solver checks.
4. Add robust particle location/interpolation on triangles and explicit wall,
   obstacle, inflow/outflow and termination policies.
5. Build transient vector/pressure datasets, time-aware GNN models and rollout
   evaluation. Scalar steady elliptic accuracy does not qualify this capability.
6. Integrate trajectory playback, field selection, probes and export into the
   interactive viewer; subsequently add heat/diffusion as a separate solver mode.
