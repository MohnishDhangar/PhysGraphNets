# Stokes velocity/traction boundary milestone

This milestone adds explicit open-boundary traction data, a physical Newtonian
stress discretization, absolute-pressure handling and boundary flux/load checks.
It extends the [full-velocity Stokes reference](stokes.md); it is not a transient
flow solver or a qualified obstacle-flow benchmark. CPU execution remains the
baseline. Viewer, video and GPU work remain deferred.

## Mathematical contract

With constant density divided out, constant kinematic viscosity nu > 0 and
outward normal n of the **fluid domain**:

    sigma(u,p) = nu (grad(u) + grad(u)^T) - p I,
    -div(sigma) = f,    div(u) = 0,
    u = g on Gamma_D,    sigma n = t on Gamma_N.

Gamma_D and Gamma_N partition every exterior edge and Gamma_D is nonempty.
Pressure and force are per unit density; integrated 2D loads are also per unit
out-of-plane depth. Tests v vanish on Gamma_D. The mixed weak form is

    2 nu integral eps(u):eps(v) - integral p div(v) - integral q div(u)
        = integral f.v + integral_Gamma_N t.v,
    eps(u) = (grad(u) + grad(u)^T)/2.

This stress formulation is documented in the
[FEniCSx flow tutorial](https://jsdokken.com/dolfinx-tutorial/chapter2/navierstokes.html)
and the [PETSc Stokes guide](https://petsc.org/release/tutorials/physics/guide_to_stokes/).
The implemented signs follow the equations above and are verified against
manufactured fields, UFL assembly and boundary force balance.

The earlier `laplacian` option uses nu grad(u):grad(v), with natural conormal
nu grad(u)n - p n. It remains available for regression compatibility. It is not
the full Newtonian traction. Select `viscous_form='symmetric_gradient'` for this
milestone's physical boundary loads. The two discrete forms can give different
solutions even though their continuous interior equations agree for constant
viscosity and exactly divergence-free velocity. Taylor-Hood velocities are only
weakly divergence-free; this distinction is retained explicitly.

### Pressure policy

- If every boundary has prescribed velocity, require compatible total flux and
  impose zero mean pressure, as in the original solver.
- If a nonempty boundary has prescribed vector traction, retain the full pressure
  space and apply **no pressure pin or mean subtraction**. Changing pressure by C
  changes traction by -C n, so the given traction sets its absolute level.
- Pure traction, slip/component-wise conditions and arbitrary mixed pressure-value
  conditions are outside this increment. A nonempty full-vector velocity boundary
  is required. No hidden zero-traction default fills a missing boundary tag.

At a shared endpoint between velocity and traction edges, the continuous velocity
DOF obeys Dirichlet data. Traction integrals still cover the entire open edge;
endpoints have zero boundary measure.

## Implementation

`mesh/structured.py` provides `tagged_unit_square_mesh` with left/right/top/bottom
tags. `fem/stokes_boundary.py` derives outward normals from the owning CCW triangle,
not the ordering of user-supplied tagged edges. It identifies all exterior vertex
and P2 edge-midpoint velocity DOFs and requires an exhaustive disjoint partition.
Normals on holes point into the hole, outward from the fluid.

`solve_stokes` accepts `dirichlet_tags` and a `tractions` mapping from tag names to
callbacks `(points, outward_normals) -> (N,2)` finite values. Zero traction must
also have an explicit callback. The stress form assembles the cross-component
velocity blocks. Edge quadrature adds the traction load, and nonzero velocity
lifting updates both momentum and continuity rows. The system remains a direct
CPU sparse saddle-point solve with no stabilization parameter.

`evaluation/stokes.py` integrates per-tag volume flux and stress loads from the
actual P2 velocity gradients and P1 pressure. Outputs distinguish:

- Net boundary flux, compared with the algebraic continuity residual.
- Physical `sigma n` acting on the fluid. The fluid's load on a solid obstacle
  is the negative of the integral on that obstacle boundary.
- Algebraic Dirichlet reactions and total body/applied-traction/reaction balance.
- Independently reconstructed stress balance, traction L2 error and per-tag force
  error against the manufactured fields. Reconstructed stress balance generally
  converges with refinement rather than being exactly zero at finite mesh size.

Laplacian-form boundary results are labeled `laplacian_conormal`; they must not
be interpreted as physical aerodynamic force measurements.

## Manufactured qualification

The affine, channel, vortex and translated-vortex fields from `StokesCase` are
used with a pressure offset of 2.5 and viscosity 0.3. Both left and right traction
orientations are tested, with exact velocity on the other three sides. Meshes
n=4,8,16 give 24 case/boundary/mesh combinations. The polynomial patches check
the absolute pressure level; the vortex cases check spatial convergence.

For the channel, u=(y(1-y),0), p=nu(1-2x)+C and f=0. Top and bottom have no slip.
For a right traction boundary:

    t_right(y) = (nu-C, nu(1-2y)),
    Q_left = -1/6,    Q_right = +1/6.

The outlet velocity is solved for, rather than prescribed. This is a **known
traction channel case**, not a zero-traction approximation to a far downstream
outlet. Even setting outlet pressure to zero leaves tangential viscous traction.
A separate exact extension-flow test exercises zero vector traction with a
nonzero constant pressure, which also catches inappropriate pressure pinning.

Other tests cover all four traction-side normals, shifted pressure with unchanged
velocity, invalid/incomplete partitions, nonfinite traction, node/edge permutation,
reversed tagged edge orientation and a manufactured patch on a polygonal hole.
The hole patch prescribes its analytical velocity at the obstacle; it is explicitly
not evidence for no-slip cylinder drag or wake accuracy.

The independent adapter uses UFL differentiation and stress assembly, identifies
boundary tags through the original mesh connectivity, and solves with PETSc/MUMPS.
It compares velocity/pressure coefficients and independently integrates each
tag's flux and load. Mixed pressure is compared without aligning away its mean.

```bash
UCX_TLS=self .fenicsx-env/bin/python scripts/run_stokes_mixed.py \
  --sizes 4 8 16 --fenicsx --output outputs/my_mixed_stokes
```

The per-command `UCX_TLS=self` setting permits serial MPI inside the restricted
execution environment. Omit `--fenicsx` for a NumPy/SciPy-only run. No dependency
installation is needed in the existing working environment.

Acceptance is declared before the study: patch field/traction errors <=1e-10,
linear residual <=1e-9, Dirichlet error <=1e-12, net flux/weak-divergence/algebraic
force balance <=1e-10. Finest-pair convergence orders must be >=2.5 for velocity
L2, >=1.8 for velocity H1, pressure L2 and divergence L2, and >=1.2 for boundary
traction L2. Finest per-tag manufactured force error must be <=0.01. External
field/flux/load differences must be <=1e-9. These are fixture-specific gates,
not universal engineering tolerances.

Each run writes a status-bearing manifest, all per-case checks/errors, versions,
source hash, revision, mesh sizes, settings and elapsed time. NPZ files preserve
boundary tags, P2 connectivity, velocity, pressure and case metadata. Outputs are
numerical data; no renderer is used.

## Next gate

The subsequent [obstacle milestone](obstacle_stokes.md) qualifies initial no-slip
cylinder/airfoil fixtures with generated-mesh audits, independent fields/loads,
boundary refinement and analytical confined-cylinder checks. Expanded domain
coverage and arbitrary-mesh import remain open. The subsequent [transient
reference](navier_stokes.md) verifies manufactured time-dependent flow. The present mixed
Stokes checks do not establish transient outlet stability, high-Reynolds flow,
moving-wall consistency, or fluid GNN accuracy.
