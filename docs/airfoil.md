# Symmetric airfoil exterior: CPU geometry milestone

The verified geometry ladder now includes a closed symmetric NACA-style airfoil
inside a circular far field. This increment solves manufactured scalar elliptic
problems with the existing CPU P1 FEM package. It adds no dependencies.

## Profile definition and coordinates

For chord fraction s in [0,1], define dimensionless half thickness

\[
T(s)=5t\left(0.2969\sqrt{s}-0.1260s-0.3516s^2
             +0.2843s^3-0.1036s^4\right).
\]

The body-frame upper and lower surfaces are
\(c(s-1/2,\pm T(s))\). Here `chord=c` is a positive length and `thickness=t`
is a nominal thickness/chord parameter, supported in [0.06,0.24]. Defaults are
c=1 and t=0.12. `center` is the **chord midpoint**, not the leading edge;
`rotation_degrees` rotates the body counterclockwise around that midpoint.
The circular far field shares this center; its radius must exceed c/2.

The base polynomial follows the [PDAS four-digit thickness formulation](https://www.pdas.com/naca456thick4.html).
Its conventional final coefficient is -0.1015 in the 5t normalization and gives
a finite trailing-edge gap. Here -0.1036 is an explicit closure modification:
the five coefficients sum to zero at s=1. The mesh has one shared trailing-edge
vertex and one shared leading-edge vertex. It is not the exact geometry of a
published NACA0012 CFD benchmark; no aerodynamic benchmark equivalence is claimed.

The analytical enclosed profile area is

\[
A_a=10tc^2\left(0.2969\frac23-\frac{0.1260}{2}
-\frac{0.3516}{3}+\frac{0.2843}{4}-\frac{0.1036}{5}\right).
\]

For the default profile this is 0.081706. It provides a reference independent
of the polygonal boundary's shoelace area.

## Mesh construction and tags

`mesh/airfoil.py` generates a conforming radial O-grid. Uniform body-frame rays
intersect the convex symmetric profile using bisection. Radial layers interpolate
between those intersections and the far-field circle. Neighboring sectors share
nodes, including the periodic seam; each cell splits into two CCW triangles.
Different sectors cannot overlap because each occupies its own angular wedge.

`nr` counts radial cells. `ntheta` counts angular sectors, must be at least 16,
and must be divisible by four, retaining exact axis-aligned endpoints before
rotation. Grading uses layer fraction (i/nr)^grading. Positive signed areas are
checked after scaling and translation, rejecting numerical collapse.

The mesh API also accepts `angular_bias` in (0,1]. Its default 1 preserves
uniform rays. Smaller values normalize (cos(theta), angular_bias*sin(theta))
to concentrate rays near the leading and trailing endpoints. The [obstacle
Stokes study](obstacle_stokes.md) uses 0.2 and reports per-tag edge lengths;
the scalar CLI and existing scalar fixtures retain their original meshes.

Boundary tags are `airfoil_upper`, `airfoil_lower`, and `farfield`. Surface tags
remain attached to the body after rotation. Inner edges run clockwise, so the
outward domain normal points into the airfoil; far-field edges run CCW.
Leading/trailing vertices belong to both adjacent surface groups.

Both curved boundaries use straight facets. Mesh size, orientation, topology,
empty-hole rendering, and boundary normals are covered by tests. Coarse angular
grids approximate the thin profile poorly; mesh refinement is essential. This
is a geometry verification mesh, without CFD boundary-layer or wake refinement.
Cambered profiles, arbitrary airfoil files, rectangular far fields, and general
Gmsh import remain outside this generator.

## Mathematical verification problem

On each polygonal exterior domain solve

\[
-\nabla\cdot(k\nabla u)+u=f,\qquad k>0.
\]

On prescribed edges use u=g; on remaining edges use
\(q=k\nabla u\cdot n\), with the **actual facet normal**. For test functions
vanishing on the Dirichlet edges,

\[
\int_\Omega k\nabla u\cdot\nabla v+uv
=\int_\Omega fv+\int_{\Gamma_N}qv.
\]

The existing geometry-independent polynomial fields are reused from the cylinder
milestone, with new tags and case identifiers in `physics/airfoil.py`:

| Case | Analytical solution | Gradient | Forcing |
|---|---|---|---|
| `quadratic` | 1+x²+y² | (2x,2y) | u-4k |
| `affine` | 1+2x-3y | (2,-3) | u |

Coordinates in these fields are global, so rotated meshes test nonradial data.
The affine patch must be reproduced to roundoff, including nonzero Dirichlet
values. The quadratic field tests convergence. References are passed only to
evaluation, independently of FEM assembly and solving.

| Boundary mode | Airfoil upper/lower | Far field |
|---|---|---|
| `dirichlet` | Prescribed u | Prescribed u |
| `airfoil_neumann` | Prescribed q | Prescribed u |
| `farfield_neumann` | Prescribed u | Prescribed q |

These are manufactured scalar data. They do not implement no-slip, aerodynamic
inflow/outflow, lift, drag, potential flow, or Navier–Stokes equations. Smooth
polynomial references do not establish regularity or convergence for arbitrary
physical data at the sharp trailing-edge corner.

## Error and geometry evidence

Metrics distinguish:

- FEM L2/H1 errors and nodal errors on the **polygonal mesh**.
- Prescribed-boundary errors and reduced algebraic residuals.
- Exact profile area versus polygonal profile area.
- Exact far-field disk area versus polygonal far-field area, plus its sagitta.
- Total mesh area versus polygonal area and ideal-domain area.

The two boundary area errors are reported separately: errors in the outer area
and hole area can partially cancel in the total domain area. Area convergence
is a geometry diagnostic, not a bound on physical solution error on a different
domain. Polynomial fields extend across both boundaries, permitting independent
error evaluation on every mesh in the sequence.

The default refinement ladder is (nr,ntheta)=(4,64),(8,128),(16,256). Both
resolutions grow together. Tests require approximately second-order L2 and
first-order H1 convergence for all modes, with decreasing profile and far-field
area errors. Rates use maximum triangle diameter, which need not halve exactly
on these coarse grids. Errors below 1e-12 do not receive fitted rates.

Additional tests check rotated/scaled affine patches, thickness extremes,
integrated quadratic flux by the divergence theorem, invalid inputs, schema-v2
archive round-trips and edge-tag reordering, CLI settings, and provenance.

## Running and artifacts

```bash
python scripts/run_airfoil.py --config configs/experiments/airfoil_local.json
python scripts/run_airfoil.py --boundary-mode farfield_neumann --convergence 4 8 16
python scripts/run_airfoil.py --case affine --rotation-degrees 17 --center 0.2 -0.3
```

On this workstation use `.pignn-venv/bin/python`. CLI flags override JSON values.
Unless specified, ntheta=angular_ratio*nr, with angular_ratio=16. Convergence
runs always use this ratio independently of single-run ntheta. The ratio must
be at least 16 and divisible by four so even the coarsest nr=1 mesh is valid.

Unique output directories default to `outputs/airfoil/`; explicit directories
must be empty. Exports include tagged schema-v2 `field.npz`, `comparison.png`,
`metrics.json`, `manifest.json`, and optional convergence CSV/PNG. The manifest
records the exact profile convention, all geometry/physics settings, mesh sizes,
quadrature, errors, Python/dependencies, CPU precision, elapsed time and code
revision/source digest. Schema-v1 and v2 readers are unchanged.

## Remaining gates

Independent FEniCSx checks passed for the default airfoil fixtures; actual
Kaggle/Colab execution remains open. See [validation](validation_20261026.md).
The user subsequently authorized starting Phase 2B alongside the external gate:
a square-domain CPU PINN is now available in `pinn.md`, and the external
comparison adapter is described in `fenicsx.md`. General mesh import, broader
airfoil families and airfoil PINN training remain future work. Initial stationary
no-slip flow is now separately qualified in [obstacle Stokes](obstacle_stokes.md);
transient and inertial airfoil flow remain open.

## Recorded local validation

The implementation validation ran the complete suite (**116 tests passed**),
the original Poisson 8/16/32 convergence demo, all three airfoil refinement
experiments, and a rotated/translated affine patch. All commands exited 0.
Python 3.14.7; CPU float64. Git revision was unavailable; source SHA-256:
`4abfe7edfd29b715de5b7cdf229eb6276b343e6e6b7e7076b5a88c16a5edaa29`. Full logs and manifests are generated under
`outputs/airfoil_validation/`. These results establish local execution only.

Finest mesh: nr=16, ntheta=256, 4,352 nodes and 8,192 triangles. Rates below
use the final 8→16 radial refinement interval.

| Mode | Relative L2 | Relative H1 | L2 rate | H1 rate |
|---|---:|---:|---:|---:|
| `dirichlet` | 0.000623949 | 0.0291358 | 2.120 | 1.076 |
| `airfoil_neumann` | 0.000817209 | 0.0286097 | 1.998 | 1.034 |
| `farfield_neumann` | 0.000616285 | 0.0291179 | 1.969 | 1.074 |

Prescribed boundary errors were zero. The rotated affine experiment had maximum
nodal error approximately 1.60e-14. The existing regression suite also covers
the preceding square, annulus, and rectangle/cylinder milestones.
