# CPU accuracy-versus-cost study

[Executed 30-trial results](validation_mixed_cost_20261028.md).

```bash
python scripts/run_pinn_cost.py --config configs/experiments/pinn_cost_cpu.json \
  --output outputs/my_cost_study
```

The runner launches a fresh process for every solver/candidate/repeat. Default
candidates are FEM meshes n=8,16,32 and Torch training budgets 300/1000 steps,
with three repeats per candidate on each of sine and variable_sine (30 trials).
The declared common targets are relative L2 <= 0.01 and relative H1 seminorm <=
0.06. A solver candidate is eligible only if **all repeats meet both targets**.
The selected candidate has the smallest median measured construction/solve cost
among the supplied eligible candidates. No eligible candidate is reported as null.

Timing scopes:

- FEM: mesh construction separately from assembly plus sparse solve.
- Torch: training including initialization and optimizer setup, plus separate
  reconstruction of the inference module. Imports and checkpoint I/O are excluded.
- Torch inference: twenty timed value-only calls on 4,096 fixed points after one
  untimed warmup. Inference is a separate result, not included in the solve-cost
  comparison. There is no FEM point-location/interpolation inference benchmark.
- Error integration, plotting and process startup are outside timed sections.
  Explicit module imports precede timing; lazy optimizer/runtime initialization
  during training remains included.
- One Torch thread and one BLAS/OpenMP thread are requested for every worker.

Memory is Linux process high-water RSS sampled before error evaluation, including
library/runtime memory and inference. A baseline high-water value is also recorded;
neither number is claimed to measure incremental tensor allocations. This runner
does not support other operating systems' differing RSS units.

FEM errors are checked at quadrature orders 5 and 7. Neural continuous errors are
evaluated on (n,order)=(16,5),(32,5),(32,7) for the same trained field. The finest
evaluation supplies the acceptance errors. Held-out residual points use a separate
assessment seed 20007, with 4,096 points. Increasing the neural evaluation mesh
refines error integration only; it does not retrain or refine the network.

All raw trial results, error sensitivity, timing samples, memory observations and
candidate selections are preserved in `cost_manifest.json`. Failures remain in
the record and disqualify the candidate. These repeated timings use one training
seed to separate timing variability from initialization variability. They do not
replace the multi-seed robustness study. Comparisons cover the sampled candidates
on this machine; they do not establish a universal best solver or matched errors
beyond satisfying the common tolerance pair. Dedicated profiling of optimizer
setup versus updates, broader candidate grids, and repeated inference across
different query sizes remain future work.
