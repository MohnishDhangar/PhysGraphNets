# Transient cylinder benchmark: DFG 2D-3

Status: implementation and qualification tooling present; scientific qualification
is pending. Complete coarse trajectories fail the unchanged published-quantity
gates. A completed run, or independent agreement on one mesh, does not qualify
continuum accuracy. Viewer, particles and learned fluid models remain later work.
See [executed development and regression evidence](validation_cylinder_flow_20261112.md).

## Formulation and boundary convention

Use the [official DFG 2D-3 specification](https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark3_re100.html):

    Omega = ([0,2.2] x [0,0.41]) \ disk(center=(0.2,0.2), radius=0.05)
    u_t + (u.grad)u - nu Delta u + grad p = 0,   div u = 0
    nu = 0.001, density = 1,   0 <= t <= 8,   u(t=0) = 0.

The cylinder and both channel walls have zero velocity. At x=0:

    u = (4 U(t) y (0.41-y) / 0.41^2, 0)
    U(t) = 1.5 sin(pi t/8).

At x=2.2 use `nu grad(u) n - p n = 0`. Here n points outward from the
fluid. This is the Laplacian conormal convention of this benchmark. It differs
from symmetric-gradient physical traction in the preceding manufactured-flow
milestone. The solver defaults to symmetric-gradient stress; this case explicitly
selects `viscous_form='laplacian'`. Traction determines absolute pressure; no
pressure pin or mean subtraction is applied.

The existing CPU float64 Taylor-Hood P2/P1 discretization uses fully implicit
advective convection, BDF2 with one BE startup step and Newton/backtracking.
The zero initial velocity satisfies the inlet at t=0 and the constrained initial
projection. No analytical interior histories or neural fields enter the solve.
Order-4 Duffy quadrature integrates the discrete degree-5 convection/test
polynomial exactly; tests compare against order 6. This lower order is specific
to the unforced cylinder case, not the nonpolynomial manufactured forcing.

## Geometry and measured quantities

The generated straight-sided mesh combines circular refinement rings and a
channel/wake point grid. Delaunay output is accepted only with all prescribed
outer edges and polygonal obstacle facets preserved. Audits check positive
triangle area/orientation, one hole, boundary tags and minimum quality. Tests
check the polygon's exact area at three resolutions and exact affine pressure
probe interpolation. This is a dedicated mesher, not general arbitrary CAD import.

Drag/lift use the cylinder's discrete momentum reactions, including unsteady and
convective terms. A separate surface-conormal integration is retained as a
refinement diagnostic. Force is on the obstacle and coefficients divide by
`0.5 U_mean_peak^2 D = 0.05`, with `U_mean_peak=1` and `D=0.1`.
The normalization never divides by the instantaneous inlet speed.

Pressure difference is signed `p(0.15,0.2)-p(0.25,0.2)`, interpolated with the
actual P1 basis. Peaks are the largest stored values, with their stored physical
times; there is no fitted phase shift or interpolated-peak correction.

### Reference pressure sign audit

The official HTML summary lists a positive final pressure difference. Its
[raw pressure archive](https://wwwold.mathematik.tu-dortmund.de/~featflow/media/dfg_bench3_2d/pressure_q2_cn_lv1-6_dt4.zip)
instead contains, in the final `pointvalues_lv6` row:

    time = 7.9996875
    p_front = -0.43873062102
    p_rear  = -0.32718923230
    front - rear = -0.11154138872.

Archive SHA-256:
`dc49ebecc647701f260646fdee02ee0bdca43d684350ac174939c9d2ead60b6f`.
The evaluator uses the signed raw value and records both the raw timestamp and
the discrepancy with the HTML table. Comparing t=8 against this last published
sample has a 0.0003125-second endpoint offset. It is a numerical reference, not
an analytical value. The raw file also lists y=0.199999 for both probes, a
1e-6 offset from the nominal y=0.2 used by our evaluator and the specification.
These small reference sampling offsets are explicit; no fitted pressure offset
or sign-insensitive comparison is applied. The sign correction does not alter tolerance magnitudes.
Original failed manifests remain preserved with their original reference values.

## Acceptance and study protocol

A single trajectory records independent execution, discrete, and benchmark
statuses. Discrete thresholds are relative Newton residual <=5e-11, prescribed
velocity/no-slip error <=1e-12, initial projection and weak divergence <=1e-10,
net flux <=1e-10 and algebraic momentum imbalance <=1e-9. Source changes during
execution invalidate its discrete acceptance.

Published gates retain 3% relative error for maximum drag, maximum lift and final
pressure difference, plus absolute peak-time error <=0.05 seconds. Only the
original case over the full [0,8] interval is assessed. Shortened trajectories
and changed physical parameters are explicitly `not_assessed`.

The predeclared complete study is
`configs/experiments/cylinder_flow_qualification.json`:

- Mesh levels 4/8/16 at fixed dt=0.005.
- Timesteps 0.02/0.01/0.005 at fixed mesh level 16.
- Full independently evolved FEniCSx trajectory on the finest mesh/timestep,
  including its initial projection, velocity/pressure fields, boundary fluxes,
  conormal loads, total reactions and obstacle reactions. Maximum absolute
  differences must be <=1e-9.
- Finest spatial and temporal pairs must agree within 3% of fixed published
  scales for the three summary quantities, 0.05 seconds for peak times, and 5%
  time-integrated RMS for drag/lift/pressure histories. Fine histories are linearly
  sampled at overlapping coarse solved times; no extrapolation is allowed.
- Finest full-history surface/reaction discrepancy must be <=5% of the published
  peak drag/lift scales. Relative instantaneous errors near zero are not used.

Every run must pass its discrete checks. Only the finest trajectory must pass
the published-quantity gates; coarse failures are retained. No study can become
`qualified` without the independent trajectory. The runner writes progress after
each completed run, rejects nonempty destinations and detects source changes.
These refinement differences are diagnostics, not certified continuum bounds.

The published channel length is fixed. Altering it changes the benchmark; an
outlet-length sensitivity experiment must be identified separately and cannot
be scored against the original benchmark as an identical physical case.

## Reproduction

Single complete run:

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 UCX_TLS=self \
  .fenicsx-env/bin/python scripts/run_cylinder_flow.py \
  --level 8 --dt .005 --fenicsx --output outputs/my_cylinder_run
```

Complete declared mesh/time qualification (substantial serial CPU runtime):

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 UCX_TLS=self \
  .fenicsx-env/bin/python scripts/qualify_cylinder_flow.py \
  --config configs/experiments/cylinder_flow_qualification.json \
  --output outputs/my_cylinder_qualification
```

Outputs include pickle-free P2/P1 trajectories with boundary tags, force-history
CSV, case/mesh/error/environment/source manifests and explicit acceptance status.
`progress.json` is replaced atomically during a run and records its stage,
completed steps, physical time and elapsed seconds. It is monitoring data, not a
restart checkpoint or an acceptance result. Progress callbacks are tested to
preserve the numerical trajectory exactly.
The complete-study CLI exits nonzero for failed qualification, including missing
independent verification. Run destinations must be fresh to preserve failures.

Once the declared mesh/time study passes, the next deliverable is triangle-based
particle location/interpolation with explicit wall, inlet and outlet policies.
