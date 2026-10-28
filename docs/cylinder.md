# Rectangle containing a cylinder: CPU geometry milestone

Implemented as a scalar diffusion/reaction verification problem. The outer
rectangle is exact; the circular obstacle is approximated by straight chords.
This is not a flow or aerodynamic model.

## Mathematical formulation

On the rectangle minus the obstacle, solve

\[
-\nabla\cdot(k\nabla u)+u=f,\qquad k>0.
\]

Dirichlet data are prescribed on a nonempty tagged boundary. On remaining
edges prescribe \(q=k\nabla u\cdot n\), where \(n\) is the actual outward
polygon-facet normal, pointing into the hole on the cylinder. The weak form is

\[
\int_\Omega k\nabla u\cdot\nabla v+uv
=\int_\Omega fv+\int_{\Gamma_N}qv.
\]

Two independent manufactured references are supplied:

- `quadratic`: \(u=1+x^2+y^2\), \(\nabla u=(2x,2y)\), \(f=u-4k\).
- `affine`: \(u=1+2x-3y\), \(\nabla u=(2,-3)\), \(f=u\).

Both extend over the plane, allowing evaluation on each polygonal domain.
The affine field is exactly representable in P1 and tests boundary elimination
and flux signs. Quadratic errors should approach second order in L2 and first
order in H1 on the default refinement sequence.

Boundary tags are `cylinder`, `bottom`, `right`, `top`, and `left`.
`dirichlet` prescribes all tags; `cylinder_neumann` prescribes the rectangle and
applies flux at the obstacle; `outer_neumann` prescribes the obstacle and applies
flux on all four rectangle sides. These are manufactured boundary data, not
physical no-slip or inflow/outflow conditions.

## Mesh and limitations

`mesh/cylinder.py` connects perimeter samples to their radial projections onto
the circle using a conforming O-grid. Every rectangle corner is retained and
the periodic seam shares nodes. `nr` counts radial cells; `nside` counts segments
on each rectangle side. The circle must lie strictly inside the rectangle;
off-center circles and radial grading are supported. Positive triangle areas
are checked, including representational collapse after coordinate translation.

This structured generator does not replace Gmsh or support arbitrary obstacles.
Very narrow gaps, extreme grading, or extreme aspect ratios can produce poor
conditioning. Default convergence claims apply to the documented geometry.

Refinement increases radial and perimeter resolution together. The hole is an
inscribed polygon, so the mesh domain has slightly more area than the ideal
domain. Metrics report polygonal and ideal areas, relative area discrepancy,
maximum cylinder chord sagitta, and solution errors separately. Error norms
are integrated on the polygonal mesh; area discrepancy alone is not an estimate
of the solution error for a different physical boundary-value problem.

## Running and artifacts

```bash
python scripts/run_cylinder.py --config configs/experiments/cylinder_local.json
python scripts/run_cylinder.py --boundary-mode outer_neumann --convergence 4 8 16
python scripts/run_cylinder.py --case affine --center 0.2 -0.1 --grading 1.3
```

CLI options override JSON settings. Default `nside=side_ratio*nr`, with
`side_ratio=2`. Every refinement uses this ratio independently of the single-run
`nside`. Output directories default to unique runs under `outputs/cylinder/`;
explicit output directories must be empty.

Artifacts: tagged schema-v2 `field.npz`, `comparison.png`, `metrics.json`,
`manifest.json`, and optional convergence CSV/PNG. The manifest records geometry,
case, boundary mode, quadrature, errors, Python/dependencies, precision, elapsed
time and code revision/source hash. Existing v1/v2 loaders are reused unchanged.

Tests cover hole topology, exact rectangle sides and outward normals, area,
nonzero affine patch tests in all boundary modes, quadratic and geometry
convergence, invalid geometries, CLI overrides, archives, and provenance.

The symmetric airfoil exterior milestone is now implemented; see [airfoil](airfoil.md). Independent FEniCSx
checks passed for the default cylinder fixtures; actual cloud execution remains
open. See [validation](validation_20261026.md). No neural
or accelerator dependencies were added.
