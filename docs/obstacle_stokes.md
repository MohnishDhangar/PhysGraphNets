# Stationary no-slip obstacle reference

This increment solves bounded, two-dimensional creeping flow around a stationary
cylinder or symmetric airfoil on the existing generated meshes. It uses the CPU
P2/P1 reference and physical stress from the [mixed Stokes milestone](stokes_mixed.md).
It adds no dependencies. [Executed qualification](validation_obstacle_stokes_20261030.md)
records the tested configurations and limitations.

## Formulation and cases

With density divided out, constant viscosity nu > 0 and the outward normal n
of the fluid domain:

    sigma = nu (grad(u) + grad(u)^T) - p I,
    -div(sigma) = 0,    div(u) = 0.

Every stationary obstacle has u=0 at all P2 boundary vertices and edge midpoints.
The stress, weak form, velocity lifting and pressure policy are unchanged from
the mixed-boundary reference. The cases use dimensionless coordinates, nu=1
and velocity scale U=0.01. Loads are per unit density and out-of-plane depth.

| Case | Domain | Outer boundary data | Pressure policy |
| --- | --- | --- | --- |
| `annular_cylinder` | Concentric circles, a=0.3 and R=2 | u=(U,0) on outer circle | Zero mean |
| `channel_cylinder` | Rectangle [-2,2] x [-0.75,0.75], centered cylinder a=0.3 | Left: u=(U(1-y²/H²),0); top/bottom: u=0; right: sigma n=0 | Set by traction |
| `airfoil` | Closed symmetric profile, chord 1, thickness parameter 0.12, circular outer radius 2 | u=(U,0) on outer circle; zero angle of attack | Zero mean |

The channel has inlet flux -4UH/3. All boundaries are straight polygonal facets
in the actual solve. The airfoil uses the [documented closed-tip profile](airfoil.md),
not a cambered NACA 4412. There is no inertial or time-derivative term.

This is not the DFG Re=20 cylinder benchmark: that problem includes nonlinear
advection, different geometry and a different outlet stress convention. Its drag
targets cannot validate these Stokes cases. See the
[primary benchmark specification](https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark1_re20.html).

## Analytical confined-cylinder reference

For the circular annulus, use the streamfunction and velocity convention

    psi = F(r) sin(theta),    u = (psi_y, -psi_x),
    F(r) = A r + B r³ + C/r + D r log(r).

The four coefficients solve F(a)=F'(a)=0 and F(R)=UR, F'(R)=U.
Writing H=F/r and k=H'/r gives

    u = (H + y² k, -xy k),
    p = nu (8 B x - 2 D x/r²),
    drag = 4 pi nu D,    lift = 0.

The pressure has zero mean by central symmetry. Tests check both circle values,
the analytical gradient against finite differences, incompressibility, momentum
balance and the integrated physical stress against the drag formula.

Numerical no-slip is imposed on polygon facets, not replaced with analytical
values on those facets. Error quadrature compares against the circular solution
smoothly extended over the small slivers inside the ideal inner circle. The
reported `analytical_errors_on_polygon` therefore includes boundary/geometry
approximation and is not a pure finite-element error on the exact annulus.

R=1.5, 2 and 3 are distinct confined problems, each with its own exact solution.
Their different drag values are physical confinement dependence. They do not
estimate a domain-independent uniform-flow limit: two-dimensional unbounded
cylinder Stokes flow has the classical Stokes paradox. See the
[Journal of Fluid Mechanics discussion](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/abs/stokes-paradox-creeping-flow-past-a-twodimensional-cylinder-in-an-infinite-domain/B68DD777CF3D756780F4B3C72B8CD275).

## Boundary and mesh checks

Velocity data can be a single callable or an exhaustive mapping from Dirichlet
tags to callables. Shared boundary corners must have consistent values; missing
tags and conflicting values fail. Traction tags remain exhaustive and explicit.
Normals derive from CCW triangle ownership, including inward-pointing normals
on holes when viewed from the fluid.

`mesh/quality.py` audits positive areas, triangle angles/scaled quality, connectedness,
complete exterior tags, oriented boundary cycles, Euler characteristic and area
agreement. It records per-tag edge lengths. These are checks for generated meshes;
the audit is not a global triangle-intersection detector or arbitrary CAD importer.

The refinement levels 4,8,16 use radial layers n and 16n angular sectors, giving
512, 2048 and 8192 triangles. Radial grading is 1.5. The airfoil experiment uses
angular bias 0.2: normalized rays (cos(theta), 0.2 sin(theta)) improve endpoint
resolution. The mesh API retains bias 1 as its backward-compatible default;
the obstacle experiment selects 0.2 explicitly in its manifest. Neither the
profile definition nor the acceptance limits change.

## Force measurements and independent verification

Two force estimates are reported separately:

1. Negative surface integral of sigma n over the obstacle.
2. Negative sum of the unreduced momentum residual at obstacle Dirichlet DOFs.
   This discrete reaction includes the pressure contribution and is the primary
   reported force. It is restricted to a closed obstacle separated from other
   boundary groups, avoiding ambiguous corner allocation.

`reaction_resistance` and `surface_resistance` divide these force vectors by nu U.
They are creeping-flow resistance measures, not conventional inertial drag
coefficients. Their difference is a refinement diagnostic, not an error bound.
Algebraic total-force balance and reconstructed surface-stress balance remain
distinct; exact discrete balance does not imply exact continuum force.

The FEniCSx adapter independently defines the boundary profiles, assembles UFL
physical stress and solves with PETSc/MUMPS on the identical mesh. It compares
P2 velocity, P1 pressure, per-tag integrated flux and surface loads, and obstacle
reactions obtained through a UFL residual functional. Full-velocity problems use
a pressure pin followed by mean removal; traction problems have no pressure pin.
Closed-obstacle total force is independent of a constant pressure shift.

This external comparison verifies the discrete implementation. Only the annular
case also has an analytical force and field reference. No analytical airfoil or
channel-cylinder drag is claimed.

## Reproduction and acceptance

```bash
UCX_TLS=self .fenicsx-env/bin/python scripts/run_obstacle_stokes.py \
  --fenicsx --output outputs/my_obstacle_stokes
```

The output directory must be empty. `UCX_TLS=self` permits serial MPI in the
restricted environment. Omit `--fenicsx` for the base NumPy/SciPy environment.
`--cases`, `--levels`, `--no-domain-study` and `--airfoil-angular-bias` control
the study. The default is nine primary solves and four annular-radius checks.

Declared checks for every solve: residual <=1e-9, maximum obstacle speed <=1e-12,
absolute net flux/U <=1e-9, algebraic force imbalance/(nu U) <=1e-8, positive drag
and optional external absolute differences <=1e-9. On the finest primary mesh:
reaction-drag change <=3%, surface/reaction force disagreement <=5%, and
absolute lift/drag <=2%. The annular case also requires analytical drag error
<=2% and velocity/pressure L2 refinement orders >=1.5. The two additional radii
require the finest analytical-drag and drag-change checks. These are declared
fixture gates, not general engineering accuracy guarantees.

Each run preserves status, parameters, checks, errors, Python/dependency versions,
revision, source hash and elapsed duration in a manifest. Tagged NPZ files retain
mesh connectivity, all P2 velocity DOFs, P1 pressure and nodal reactions. Generated
artifacts remain under ignored `outputs/` directories.

## Remaining work

The subsequent [transient increment](navier_stokes.md) implements manufactured
time-dependent incompressible Navier-Stokes with timestep and spatial convergence.
The next numerical gate is a precisely specified transient cylinder benchmark.
Channel outlet-length and airfoil outer-radius sensitivity, broader shape/flow
parameters and arbitrary-mesh import remain open. Then come triangle-based
particle transport, transient datasets and fluid-model rollout qualification.
The existing scalar GNN does not predict these vector/pressure fields. Viewer,
final-video and GPU work remain deferred.
