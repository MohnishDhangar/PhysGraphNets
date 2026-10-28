"""Full-batch Adam with fixed collocation and pickle-free complete checkpoints."""
from dataclasses import dataclass, asdict
import json
from pathlib import Path

import numpy as np

from physgraph.models.pinn import TanhMLP
from physgraph.physics.pinn import get_pinn_case, CASE_NAMES
from physgraph.losses.pinn import loss, loss_gradient
from physgraph.io.pinn_provenance import training_fingerprint


@dataclass(frozen=True)
class TrainingConfig:
    case: str = 'sine'
    width: int = 8
    depth: int = 2
    collocation: int = 128
    learning_rate: float = .003
    seed: int = 7
    neumann_points: int = 32
    neumann_weight: float = 1.

    def __post_init__(self):
        TanhMLP(self.width, self.depth)
        if self.case not in CASE_NAMES:
            raise ValueError('unsupported PINN case')
        for name, value, minimum in (('collocation', self.collocation, 4), ('seed', self.seed, 0), ('neumann_points', self.neumann_points, 2)):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}')
        if not np.isfinite(self.neumann_weight) or self.neumann_weight <= 0:
            raise ValueError('neumann_weight must be positive and finite')
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError('learning_rate must be positive and finite')


@dataclass
class TrainingState:
    parameters: np.ndarray
    first_moment: np.ndarray
    second_moment: np.ndarray
    points: np.ndarray
    losses: list[float]
    step: int
    rng_state: dict


def save_checkpoint(path, config, state, *, backend='numpy_complex_step_adam', runtime=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {'schema_version': 2, 'backend': backend, 'training_sha256': training_fingerprint(),
                'config': asdict(config), 'step': state.step, 'rng_state': state.rng_state,
                'adam': {'beta1': .9, 'beta2': .999, 'epsilon': 1e-8}}
    if runtime is not None:
        metadata['runtime'] = runtime
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, parameters=state.parameters, first_moment=state.first_moment,
            second_moment=state.second_moment, points=state.points, losses=np.asarray(state.losses),
            metadata=np.array(json.dumps(metadata, allow_nan=False)))
    temporary.replace(path)


def load_checkpoint(path, config, *, backend='numpy_complex_step_adam', expected_runtime=None):
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata'].item()))
        arrays = {key: archive[key].copy() for key in ('parameters', 'first_moment', 'second_moment', 'points', 'losses')}
    if (metadata.get('schema_version') != 2 or metadata.get('backend') != backend
            or metadata.get('config') != asdict(config)
            or metadata.get('adam') != {'beta1': .9, 'beta2': .999, 'epsilon': 1e-8}):
        raise ValueError('checkpoint schema, backend, optimizer or training configuration does not match')
    if metadata.get('training_sha256') != training_fingerprint():
        raise ValueError('checkpoint training source fingerprint does not match; start a new experiment')
    if expected_runtime is not None and metadata.get('runtime') != expected_runtime:
        raise ValueError('checkpoint PyTorch version or CPU runtime settings do not match')
    step = metadata['step']
    count = TanhMLP(config.width, config.depth).parameter_count
    expected = {'parameters': (count,), 'first_moment': (count,), 'second_moment': (count,),
                'points': (config.collocation, 2), 'losses': (step+1,) if isinstance(step, int) else ()}
    if (isinstance(step, bool) or not isinstance(step, int) or step < 0
            or any(a.shape != expected[key] or a.dtype != np.float64 or not np.isfinite(a).all() for key, a in arrays.items())
            or np.any(arrays['second_moment'] < 0) or np.any(arrays['losses'] < 0)
            or not get_pinn_case(config.case).domain.contains(arrays['points'], strict=True)):
        raise ValueError('checkpoint state is malformed or nonfinite')
    rng = np.random.default_rng()
    rng.bit_generator.state = metadata['rng_state']
    return TrainingState(arrays['parameters'], arrays['first_moment'], arrays['second_moment'],
                         arrays['points'], arrays['losses'].tolist(), step, metadata['rng_state'])


def train(config: TrainingConfig, steps: int, *, checkpoint=None, resume=None,
          checkpoint_every: int = 100, progress=None) -> TrainingState:
    """steps is the total target Adam step, including steps completed on resume."""
    for name, value, minimum in (('steps', steps, 0), ('checkpoint_every', checkpoint_every, 1)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    loss_options = {'neumann_points': config.neumann_points, 'neumann_weight': config.neumann_weight}
    model, case = TanhMLP(config.width, config.depth), get_pinn_case(config.case)
    if resume is None:
        rng = np.random.default_rng(config.seed)
        parameters = model.initialize(rng)
        points = case.domain.sample(rng, config.collocation)
        state = TrainingState(parameters, np.zeros_like(parameters), np.zeros_like(parameters),
            points, [float(loss(model, parameters, points, case, **loss_options))], 0, rng.bit_generator.state)
    else:
        state = load_checkpoint(resume, config)
    if steps < state.step:
        raise ValueError('target steps cannot precede the checkpoint step')
    for step in range(state.step+1, steps+1):
        gradient = loss_gradient(model, state.parameters, state.points, case, **loss_options)
        state.first_moment = .9*state.first_moment+.1*gradient
        state.second_moment = .999*state.second_moment+.001*gradient**2
        state.parameters -= config.learning_rate*(state.first_moment/(1-.9**step))/(np.sqrt(state.second_moment/(1-.999**step))+1e-8)
        current_loss = float(loss(model, state.parameters, state.points, case, **loss_options))
        if not np.isfinite(current_loss):
            raise FloatingPointError('nonfinite training loss')
        state.losses.append(current_loss)
        state.step = step
        if step % checkpoint_every == 0:
            if checkpoint is not None:
                save_checkpoint(checkpoint, config, state)
            if progress is not None:
                progress(step, current_loss)
    if checkpoint is not None:
        save_checkpoint(checkpoint, config, state)
    return state
