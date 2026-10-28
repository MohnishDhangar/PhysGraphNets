`cost_worker.py` executes one isolated FEM or CPU PyTorch trial, with explicit
timing scopes, Linux peak RSS and independent error evaluation. The package
runner is `experiments/pinn_cost.py`; see `docs/cost_study.md`. GPU profiling and
custom kernels remain deferred.
