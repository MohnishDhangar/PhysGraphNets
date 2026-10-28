Initial CPU full-batch Adam with fixed collocation and deterministic NPZ checkpoint/resume: `pinn.py`. Graph datasets and training remain future work.
Optional native PyTorch Adam with CPU float64, scoped deterministic settings and compatible NPZ storage: `torch_pinn.py`. Resume stays within the originating backend.
See `docs/pinn.md` for scope and evidence.
