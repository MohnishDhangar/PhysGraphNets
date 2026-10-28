# Annulus geometry milestone

The numerical geometry ladder now includes a radial/angular mesh surrounding a
circular hole. This is a two-dimensional cylinder cross-section with a circular
outer boundary, solving scalar steady diffusion. The [airfoil geometry milestone](airfoil.md) is also implemented. Aerodynamic
flow remains a separate future milestone. This increment adds no dependencies.

## Analytical problem

Let the ideal annulus be \(a<r<b\), where
\(r=\|x-x_c\|\). For a constant scalar diffusivity \(k>0\), solve

\[
-\nabla\cdot(k\nabla u)=0,\qquad u(a)=u_i,\qquad u(b)=u_o.
\]

The radial Laplacian is \(u''(r)+u'(r)/r\), giving

\[
B=\frac{u_o-u_i}{\log(b/a)},\qquad
u(r)=u_i+B\log(r/a),\qquad
\nabla u=B\frac{x-x_c}{r^2}.
\]

Defaults are \(a=0.5\), \(b=1\), \(u_i=1\), \(u_o=2\), \(k=1\).
Both prescribed values are nonzero. The logarithmic reference is smooth on
\(r>0\), including the small differences between circular and polygonal domains.

Three boundary variants use the same field:

| `boundary_mode` | Inner boundary | Outer boundary |
|---|---|---|
| `dirichlet` | Prescribed u | Prescribed u |
| `inner_neumann` | Prescribed q | Prescribed u |
| `outer_neumann` | Prescribed u | Prescribed q |

The convention is \(q=k\nabla u\cdot n\), which enters the weak-form load
with a **positive** boundary integral. On ideal circles, \(q=-kB/a\) at the
inner boundary and \(q=kB/b\) at the outer boundary. The outward normal on
the inner boundary points **into the hole**. Physical diffusive flux is \(-q\).

## Mesh and boundary tags

`mesh/annulus.py` generates \(n_r+1\) radial rings and \(n_\theta\) angular
vertices per ring, with no duplicate vertex at \(2\pi\). Adjacent angular cells
share the seam. Each cell becomes two CCW P1 triangles:

- Nodes: \((n_r+1)n_\theta\).
- Triangles: \(2n_r n_\theta\).
- Exterior edges: \(2n_\theta\), split equally between `inner` and `outer`.
- Optional radial grading: \(r_i=a+(b-a)(i/n_r)^p\), \(p>0\).

`Mesh2D.boundary` marks the geometric exterior. Optional `boundary_edges` and
`boundary_edge_tags` arrays attach names to each exterior edge exactly once.
`BoundaryConditions.dirichlet_tags` selects prescribed edges using these names.
Unselected exterior edges require Neumann data. Existing midpoint selectors and
untagged meshes remain supported. Tag lookup ignores stored edge direction;
assembly derives outward normals from the CCW triangle topology.

## Curved geometry versus the discrete verification domain

These are **straight-sided P1 triangles**. The circular boundaries are replaced
by regular polygons whose vertices lie on the circles. The inner polygon is
slightly smaller than the ideal circular hole. No triangle fills the polygonal
hole, but a thin chordal region can lie inside the ideal radius a.

The mesh's exact polygonal area is

\[
|\Omega_h|=\tfrac12 n_\theta\sin(2\pi/n_\theta)(b^2-a^2),
\]

whereas the circular area is \(\pi(b^2-a^2)\). For a radius R, the maximum
arc-to-chord distance is \(R[1-\cos(\pi/n_\theta)]\). Both area discrepancy
and boundary displacement decrease quadratically with angular refinement.

Solution error norms are integrated **on the polygonal mesh** against the
analytical logarithmic field extended to that domain. They are not exact norms
over the ideal circular domain. Dirichlet data is the nodal interpolation of the
analytical trace; this interpolation is constant along each ring's boundary edges.
The reference's trace along a chord is generally not constant between vertices.

For mixed tests, Neumann quadrature evaluates

\[
q(x)=kB\frac{(x-x_c)\cdot n_h}{\|x-x_c\|^2}
\]

using the **actual straight-edge normal** \(n_h\). It does not substitute the
constant circular-boundary flux. Thus the mixed tests verify the weak formulation
on the computational polygon, including inner-boundary sign and orientation.

The analytical integrated q is \(-2\pi kB\) on the inner loop and
\(+2\pi kB\) on the outer loop, also for these polygonal contours. Tests compare
assembled Neumann load sums to those independently specified values.

## Running and artifacts

With the project environment active:

```bash
python scripts/run_annulus.py --config configs/experiments/annulus_local.json
python scripts/run_annulus.py --boundary-mode inner_neumann --convergence 4 8 16
python scripts/run_annulus.py --boundary-mode outer_neumann --convergence 4 8 16
```

On this workstation, `.pignn-venv/bin/python` can replace `python` directly.
These commands use the same installed package on local CPU, Kaggle or Colab.

`--nr` counts radial cells; `--ntheta` counts angular sectors. With no explicit
`--ntheta`, it is `angular_ratio * nr` (default ratio 8). Refinement studies
always set `ntheta=angular_ratio*nr` for each requested radial size, even if the
single-run angular count was specified explicitly. Refining radial cells alone
leaves both angular discretization and the curved-boundary approximation fixed.

Runs create unique directories under `outputs/annulus/`; an explicit `--output`
directory must be empty. Outputs include:

- `field.npz`: schema v2 with tagged exterior edges and FEM/reference fields.
- `comparison.png`: FEM, reference, signed nodal error, and mesh with boundary tags.
- `metrics.json`: L2/H1, nodal/boundary errors, residuals, mesh area and geometry discrepancies.
- `convergence.csv`, `convergence.png`: when refinement is requested; rates use
  the maximum element diameter and report both solution and geometry-area rates.
- `manifest.json`: complete radii, center, grading, boundary mode, values,
  diffusivity, quadrature, mesh sizes, errors, environment and source provenance.

Schema v1 remains readable and is still written for untagged meshes. Schema v2
adds edge tags; malformed or incomplete tag coverage is rejected during loading.

## Acceptance and remaining work

Tests cover positive areas, the polygonal area formula, Euler characteristic 0
for a one-hole domain, shared seam connectivity, an empty hole in the plotting
triangulation, inward-facing inner normals, both mixed flux signs, a nonradial
affine patch test, nonzero/constant boundary values, and v2 export/reload.

The default refinement ladder is `(nr, ntheta) = (4,32), (8,64), (16,128)`.
For the nonconstant log case, acceptance requires near-second-order L2 and
near-first-order H1 convergence on both intervals for all boundary modes.
For a constant reference, use absolute H1 error because its reference seminorm
is zero. Small algebraic residual alone is not an accuracy criterion.

The rectangular outer domain containing a cylinder is now implemented; see
[cylinder milestone](cylinder.md). The [symmetric airfoil exterior](airfoil.md) also has named boundary groups
and controlled refinement.
General mesh import remains pending. Independent FEniCSx checks passed for the
default annulus fixtures; see [validation](validation_20261026.md). This
geometry increment introduces neither Navier–Stokes physics nor neural training.

