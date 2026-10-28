"""Native torch.optim.Adam training with deterministic, pickle-free CPU resume."""
import numpy as np
import torch

from physgraph.core.torch_runtime import cpu_runtime, runtime_metadata
from physgraph.models.pinn import TanhMLP
from physgraph.models.torch_pinn import TorchTanhMLP
from physgraph.physics.pinn import get_pinn_case
from physgraph.physics.torch_pinn import sample_physics
from physgraph.losses.torch_pinn import loss
from physgraph.training.pinn import TrainingConfig, TrainingState, load_checkpoint as load_state, save_checkpoint

BACKEND = 'torch_autograd_adam'


def load_checkpoint(path, config, *, threads=1):
    return load_state(path, config, backend=BACKEND, expected_runtime=runtime_metadata(threads))


def train(config: TrainingConfig, steps: int, *, checkpoint=None, resume=None,
          checkpoint_every: int = 100, progress=None, threads: int = 1) -> TrainingState:
    for name, value, minimum in (('steps', steps, 0), ('checkpoint_every', checkpoint_every, 1)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    runtime = runtime_metadata(threads)
    case = get_pinn_case(config.case)
    if resume is None:
        # Identical draws/layout enable direct comparisons with the NumPy model.
        rng = np.random.default_rng(config.seed)
        parameters = TanhMLP(config.width, config.depth).initialize(rng)
        points = case.domain.sample(rng, config.collocation)
        state = TrainingState(parameters, np.zeros_like(parameters), np.zeros_like(parameters),
                              points, [], 0, rng.bit_generator.state)
    else:
        state = load_checkpoint(resume, config, threads=threads)
    if steps < state.step:
        raise ValueError('target steps cannot precede the checkpoint step')
    with cpu_runtime(threads):
        model = TorchTanhMLP(config.width, config.depth, state.parameters)
        data = sample_physics(state.points, case, neumann_points=config.neumann_points, neumann_weight=config.neumann_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
            betas=(.9, .999), eps=1e-8, weight_decay=0., amsgrad=False, foreach=False, fused=False)
        # A single flat parameter makes the complete Adam state portable to NPZ.
        opt = optimizer.state[model.coefficients]
        opt.update(step=torch.tensor(float(state.step), dtype=torch.float64, device='cpu'),
                   exp_avg=torch.tensor(state.first_moment, dtype=torch.float64, device='cpu'),
                   exp_avg_sq=torch.tensor(state.second_moment, dtype=torch.float64, device='cpu'))
        if resume is None:
            state.losses.append(float(loss(model, data, create_graph=False).detach()))

        def snapshot():
            state.parameters = model.coefficients.detach().numpy().copy()
            state.first_moment = opt['exp_avg'].detach().numpy().copy()
            state.second_moment = opt['exp_avg_sq'].detach().numpy().copy()
            if int(opt['step'].item()) != state.step:
                raise RuntimeError('Adam and checkpoint step counters disagree')
            if checkpoint is not None:
                save_checkpoint(checkpoint, config, state, backend=BACKEND, runtime=runtime)

        for step in range(state.step+1, steps+1):
            optimizer.zero_grad(set_to_none=True)
            objective = loss(model, data)
            objective.backward()
            if not torch.isfinite(objective) or not torch.isfinite(model.coefficients.grad).all():
                raise FloatingPointError('nonfinite PyTorch PINN loss or parameter gradient')
            optimizer.step()
            current_loss = float(loss(model, data, create_graph=False).detach())
            if not np.isfinite(current_loss):
                raise FloatingPointError('nonfinite PyTorch training loss')
            state.losses.append(current_loss)
            state.step = step
            if step % checkpoint_every == 0:
                snapshot()
                if progress is not None:
                    progress(step, current_loss)
        snapshot()
    return state
