# Mixed-boundary square PINNs

[Executed validation and experiment results](validation_mixed_cost_20261028.md).

The square cases `mixed_sine_right` and `mixed_sine_left` extend both NumPy and
CPU PyTorch training. The existing PDE remains

\[
-\nabla\cdot(k\nabla u)+u=f,\qquad k=1+x+y.
\]

For the right-edge case, the manufactured reference is
\(u=1+x+2y+\sin(\pi x/2)\sin(\pi y)\). The left-edge case replaces x inside the
sine with 1-x. If s denotes that sine product, the forcing is
\(f=(5\pi^2/4)ks-u_x-u_y+u\). References are used only for evaluation and
manufacturing the physical input; training receives forcing, coefficients,
boundary lifting and prescribed flux.

Three edges impose Dirichlet values. The remaining vertical edge imposes
\(q=k\nabla u\cdot n\): q=k on the right and q=-k on the left. This is the
outward diffusive derivative, with the same sign convention as the FEM solver.
The physical flux -k grad(u).n would have the opposite sign.

The network fields are

\[
u_\theta=1+x+2y+4xy(1-y)N_\theta \quad\text{(right Neumann)},
\]
\[
u_\theta=1+x+2y+4(1-x)y(1-y)N_\theta \quad\text{(left Neumann)}.
\]

These masks impose only the three Dirichlet edges. The Neumann edge remains
free. At the two junction corners, Dirichlet values are prescribed, consistent
with the FEM edge partition. The loss is interior normalized residual MSE plus
`neumann_weight` times the normalized squared flux error integrated on the
Neumann edge using `neumann_points` Gauss nodes. Both defaults are explicit:
weight 1, order 32. Each term is normalized by max(mean(target²),1).

NumPy uses analytical mask derivatives and complex-step parameter gradients.
Torch differentiates the complete constrained field; flux training retains the
first-derivative graph for parameter differentiation. Independent dense edge
midpoints evaluate RMS and maximum Neumann flux error. `max_boundary_error`
measures Dirichlet edges only; it does not include Neumann solution values.

```bash
python scripts/run_pinn.py --config configs/experiments/pinn_mixed_cpu.json
python scripts/run_pinn.py --config configs/experiments/pinn_mixed_cpu.json --case mixed_sine_left
python scripts/compare_fenicsx.py --cases mixed_sine_right mixed_sine_left --resolutions 4 8
```

Use the installed `.fenicsx-env` for Torch and external comparisons. `--backend
numpy` also supports these cases. `--neumann-points` and `--neumann-weight` are
checkpointed training settings. A change to the training source/configuration
invalidates previous checkpoint compatibility, as documented in `reliability.md`.

Tests cover outward signs, finite-difference divergence, FEM convergence, mask
derivatives, exact Dirichlet data, NumPy/Torch loss-gradient parity, training and
flux-error reduction, exact resume, and independently generated UFL forcing and
flux. This implementation does not yet support arbitrary Neumann edge subsets,
irregular neural domains, or pure Neumann problems.
