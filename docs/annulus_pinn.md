# Annulus PINN and reusable domain interfaces

[Executed experiments and validation](validation_annulus_graph.md).

The `annulus_log` case fits the verified harmonic problem -Delta u=0 on
0.5 < r < 1, with u=1 on the inner circle and u=2 on the outer circle.
The analytical reference is 1+log(r/0.5)/log(2), and remains confined to the
manufactured-case and evaluation paths. Both NumPy and CPU Torch are supported.

## Hard boundary constraint

With a=0.5, b=1 and t=(r²-a²)/(b²-a²), the prediction is

\[
u_\theta=1+t+4t(1-t)N_\theta(x,y).
\]

The lifting 1+t uses the two boundary values. It is a polynomial in Cartesian
coordinates and is **not** the logarithmic interior solution. The mask vanishes
on both exact circles, imposing nonzero Dirichlet data to roundoff. The network
coordinates are mapped from the domain bounding box to [0,1]² before the existing
MLP maps its inputs to [-1,1]². Gradients and Laplacians include this coordinate
map and all mask/lifting derivatives.

`core/domains.py` supplies membership, seeded sampling, boundary sampling, meshes
and physical-domain quadrature. `models/constraints.py` supplies boundary lifting,
mask values and independent analytical derivatives. The same value operations
remain differentiable under Torch. PDE coefficient gradients may now be constant
vectors or callables returning one finite vector per point.

Interior samples use uniform r² and uniform angle, giving **uniform area**.
The hole and exterior are excluded. Saved collocation points are checked against
the case's domain on resume; square-only coordinate checks are no longer assumed.
The domain and constraint source files participate in checkpoint compatibility.

## Physical versus polygonal domain errors

The PINN is trained on the exact circular annulus. Primary neural L2/H1 errors
use Gauss quadrature in radius (8 times the requested order) and a periodic
trapezoidal rule in angle (64 times the requested order), including the r Jacobian.
Boundary checks use exact-circle points. Held-out residuals use a separate
area-uniform sample.

The FEM mesh has straight chordal facets. Its errors therefore integrate over
the polygonal domain. Neural metrics prefixed `polygon_` evaluate the same
network on that polygonal domain for comparison with FEM. The harmonic reference
extends smoothly across the small chordal differences. Both areas and the
relative area discrepancy are recorded explicitly. Primary circular neural
errors must not be silently substituted into a polygon-domain FEM comparison.

The cost-study runner currently rejects curved cases until a suitable common
geometry/accuracy protocol is established. Square benchmarks retain their scope.

## Running

```bash
"$HOME/.local/bin/micromamba" run -p "$PWD/.fenicsx-env" \
  python scripts/run_pinn.py --config configs/experiments/pinn_annulus_cpu.json
```

For this case, `--nx` is the number of radial cells; the FEM mesh uses 8*nx angular
sectors. Model training samples and true-domain error quadrature do not depend on
this visualization/reference mesh. `fem_reference.npz` retains the true FEM
solution and boundary tags; `comparison.npz` stores neural predictions separately.

The initial config uses 3,000 steps. The recorded seed-7 run missed its residual
acceptance threshold despite low solution error. Extending the same checkpoint
to 6,000 total steps passed the unchanged thresholds:

```bash
python scripts/run_pinn.py --config configs/experiments/pinn_annulus_cpu.json \
  --resume outputs/annulus_pinn_20261028/seed7/checkpoint.npz --steps 6000
```

This is an adaptive training-budget extension on one instance/seed, not an
independently selected performance optimum. The initial failure is retained.
The existing `annulus_dirichlet` FEniCSx fixture verifies this same physical FEM
reference; no new PDE family or external dependency is introduced.

The public runner currently exposes the fixed centered annulus case. Domain and
constraint components support translated/scaled annuli and are tested for those
coordinates, but configurable geometry and annulus mixed neural boundaries
remain future extensions.
