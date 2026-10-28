"""Independent derivative parity, optimizer replay and scientific artifacts."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip('torch', reason='optional CPU PyTorch backend is not installed')

from physgraph.core.torch_runtime import cpu_runtime, runtime_metadata
from physgraph.evaluation.pinn import evaluate_pinn as evaluate_numpy
from physgraph.evaluation.torch_pinn import evaluate_pinn
from physgraph.io.results import load_field
from physgraph.losses.pinn import loss as numpy_loss, loss_gradient, strong_residual as numpy_residual
from physgraph.losses.torch_pinn import loss, strong_residual
from physgraph.mesh.structured import unit_square_mesh
from physgraph.models.pinn import TanhMLP, constrained_field
from physgraph.models.torch_pinn import TorchTanhMLP, constrained_value, field_derivatives
from physgraph.physics.pinn import DIRICHLET_CASE_NAMES as CASE_NAMES, get_pinn_case
from physgraph.physics.torch_pinn import sample_physics
from physgraph.training.pinn import TrainingConfig, train as train_numpy, load_checkpoint as load_numpy
from physgraph.training.torch_pinn import train, load_checkpoint


@pytest.fixture(scope='module', autouse=True)
def single_thread():
    with cpu_runtime(1):
        yield


@pytest.mark.parametrize('name', CASE_NAMES)
@pytest.mark.parametrize('width,depth', [(1, 1), (3, 2), (4, 3)])
def test_autograd_field_residual_and_parameter_gradient_match_independent_numpy(name, width, depth):
    architecture = TanhMLP(width, depth)
    parameters = architecture.initialize(np.random.default_rng(21))
    points = np.random.default_rng(22).random((17, 2))
    case = get_pinn_case(name)
    model, data = TorchTanhMLP(width, depth, parameters), sample_physics(points, case)
    expected = constrained_field(architecture, parameters, points, case.lifting)
    actual = field_derivatives(model, data.points, case.lifting)
    for reference, result in zip(expected, actual):
        np.testing.assert_allclose(result.detach().numpy(), reference, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(strong_residual(model, data).detach().numpy(),
        numpy_residual(architecture, parameters, points, case), rtol=2e-12, atol=2e-12)
    objective = loss(model, data)
    assert objective.item() == pytest.approx(numpy_loss(architecture, parameters, points, case), rel=2e-12)
    objective.backward()
    np.testing.assert_allclose(model.coefficients.grad.numpy(),
        loss_gradient(architecture, parameters, points, case), rtol=3e-11, atol=3e-12)
    assert model.coefficients.dtype == torch.float64 and model.coefficients.device.type == 'cpu'


@pytest.mark.parametrize('name', CASE_NAMES)
def test_hard_boundary_matches_all_four_edges(name):
    architecture, case = TanhMLP(3, 2), get_pinn_case(name)
    model = TorchTanhMLP(3, 2, architecture.initialize(np.random.default_rng(19)))
    t = np.linspace(0, 1, 29)
    points = np.vstack((np.column_stack((t, t*0)), np.column_stack((t, t*0+1)),
                        np.column_stack((t*0, t)), np.column_stack((t*0+1, t))))
    value = constrained_value(model, torch.tensor(points, dtype=torch.float64), case.lifting)
    np.testing.assert_allclose(value.detach().numpy(), case.problem.boundary.dirichlet(points), atol=5e-16, rtol=0)


@pytest.mark.parametrize('name', ['affine', 'affine_diffusion'])
def test_exact_affine_solution_has_zero_residual_and_continuous_errors(name):
    architecture, case = TanhMLP(3, 2), get_pinn_case(name)
    parameters = np.zeros(architecture.parameter_count)
    model = TorchTanhMLP(3, 2, parameters)
    data = sample_physics(np.random.default_rng(20).random((31, 2)), case)
    objective = loss(model, data)
    objective.backward()  # zero second derivatives still permit parameter differentiation
    assert objective.item() < 1e-27
    assert torch.isfinite(model.coefficients.grad).all()
    _, errors = evaluate_pinn(unit_square_mesh(4), architecture, parameters, case)
    for key in ('absolute_l2', 'absolute_h1_seminorm', 'holdout_residual_rms', 'max_boundary_error'):
        assert errors[key] < 1e-14


def test_physics_sampling_never_reads_analytical_reference():
    case = replace(get_pinn_case('variable_sine'), reference=None)
    data = sample_physics(np.array([[.2, .3], [.4, .8]]), case)
    architecture = TanhMLP(3, 1)
    model = TorchTanhMLP(3, 1, architecture.initialize(np.random.default_rng(1)))
    loss(model, data).backward()
    assert torch.isfinite(model.coefficients.grad).all()


def test_optimizer_matches_numpy_trajectory():
    config = TrainingConfig(case='variable_sine', width=3, depth=1, collocation=16)
    expected, actual = train_numpy(config, 15), train(config, 15)
    np.testing.assert_array_equal(expected.points, actual.points)
    for key in ('parameters', 'first_moment', 'second_moment', 'losses'):
        np.testing.assert_allclose(getattr(actual, key), getattr(expected, key), rtol=5e-11, atol=5e-13)


def test_training_reduces_loss_and_independently_evaluated_error():
    config = TrainingConfig(width=4, depth=1, collocation=48, learning_rate=.01)
    initial, trained = train(config, 0), train(config, 200)
    architecture, case, mesh = TanhMLP(4, 1), get_pinn_case('sine'), unit_square_mesh(6)
    _, before = evaluate_pinn(mesh, architecture, initial.parameters, case)
    prediction, after = evaluate_pinn(mesh, architecture, trained.parameters, case)
    reference_prediction, reference_metrics = evaluate_numpy(mesh, architecture, trained.parameters, case)
    np.testing.assert_allclose(prediction, reference_prediction, rtol=1e-12, atol=1e-14)
    for key in after:
        assert after[key] == pytest.approx(reference_metrics[key], rel=2e-11, abs=1e-13)
    assert trained.losses[-1] < initial.losses[0]*.1
    assert after['relative_l2'] < before['relative_l2']*.2
    assert after['holdout_residual_rms'] < before['holdout_residual_rms']*.3


def test_checkpoint_resume_replays_exact_state_and_rejects_incompatible_checkpoints(tmp_path):
    config = TrainingConfig(width=3, depth=1, collocation=16)
    complete = train(config, 30)
    path = tmp_path/'checkpoint.npz'
    train(config, 12, checkpoint=path, checkpoint_every=5)
    resumed = train(config, 30, resume=path)
    for key in ('parameters', 'first_moment', 'second_moment', 'points', 'losses'):
        np.testing.assert_array_equal(getattr(complete, key), getattr(resumed, key))
    assert complete.rng_state == resumed.rng_state
    with pytest.raises(ValueError, match='configuration'):
        load_checkpoint(path, replace(config, seed=8))
    with pytest.raises(ValueError, match='precede'):
        train(config, 2, resume=path)
    with pytest.raises(ValueError, match='runtime'):
        load_checkpoint(path, config, threads=2)
    with pytest.raises(ValueError, match='backend'):
        load_numpy(path, config)
    numpy_path = tmp_path/'numpy.npz'
    train_numpy(config, 1, checkpoint=numpy_path)
    with pytest.raises(ValueError, match='backend'):
        train(config, 3, resume=numpy_path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    metadata = json.loads(arrays['metadata'].item())
    metadata['runtime']['torch_version'] = 'incompatible'
    np.savez_compressed(tmp_path/'version.npz', **{**arrays, 'metadata': np.array(json.dumps(metadata))})
    with pytest.raises(ValueError, match='version'):
        load_checkpoint(tmp_path/'version.npz', config)
    arrays['second_moment'][0] = -1
    np.savez_compressed(tmp_path/'broken.npz', **arrays)
    with pytest.raises(ValueError, match='malformed'):
        load_checkpoint(tmp_path/'broken.npz', config)


def test_runtime_settings_and_global_rng_are_restored_even_on_failure():
    rng_state = torch.get_rng_state().clone()
    numpy_rng = np.random.get_state()
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    dtype = torch.get_default_dtype()
    train(TrainingConfig(width=3, depth=1, collocation=16), 2, threads=2)
    with pytest.raises(RuntimeError, match='injected'):
        with cpu_runtime(2):
            assert torch.get_num_threads() == 2
            assert torch.are_deterministic_algorithms_enabled()
            raise RuntimeError('injected failure')
    assert torch.equal(rng_state, torch.get_rng_state())
    np.testing.assert_array_equal(numpy_rng[1], np.random.get_state()[1])
    assert torch.get_num_threads() == threads
    assert torch.are_deterministic_algorithms_enabled() == deterministic
    assert torch.is_deterministic_algorithms_warn_only_enabled() == warn
    assert torch.get_default_dtype() == dtype


def test_cpu_float64_and_valid_thread_count_are_required():
    architecture = TanhMLP(3, 1)
    model = TorchTanhMLP(3, 1, np.zeros(architecture.parameter_count))
    with pytest.raises(ValueError, match='float64'):
        model(torch.ones((3, 2), dtype=torch.float32))
    for value in (True, 0, 1.5):
        with pytest.raises(ValueError, match='positive integer'):
            runtime_metadata(value)


def test_torch_cli_resume_and_separate_reference_artifacts(tmp_path):
    config = tmp_path/'config.json'
    config.write_text(json.dumps({'backend': 'torch', 'case': 'affine_diffusion', 'width': 3,
        'depth': 1, 'collocation': 16, 'steps': 2, 'nx': 4}))
    root = Path(__file__).resolve().parents[1]
    first, resumed, complete = (tmp_path/name for name in ('first', 'resumed', 'complete'))
    for output, extra in ((first, []), (resumed, ['--steps', '4', '--resume', str(first/'checkpoint.npz')]),
                          (complete, ['--steps', '4'])):
        subprocess.run([sys.executable, 'scripts/run_pinn.py', '--config', str(config),
            '--output', str(output), *extra], cwd=root, capture_output=True, text=True, check=True)
    manifest = json.loads((resumed/'manifest.json').read_text())
    assert manifest['config']['backend'] == 'torch_autograd_adam'
    assert manifest['config']['start_step'] == 2 and manifest['config']['steps'] == 4
    assert len(manifest['config']['resume_sha256']) == 64
    assert manifest['config']['torch_runtime'] == runtime_metadata()
    assert manifest['dependencies']['torch'] == str(torch.__version__)
    assert manifest['metrics']['status'] == 'completed'
    assert manifest['metrics']['fem']['max_nodal_error'] < 1e-12
    assert manifest['metrics']['pinn']['max_nodal_error'] > 1e-6
    fem, _ = load_field(resumed/'fem_reference.npz')
    with np.load(resumed/'comparison.npz', allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive['fem_solution'], fem.solution)
        assert not np.array_equal(archive['fem_solution'], archive['pinn_prediction'])
    with np.load(resumed/'checkpoint.npz', allow_pickle=False) as a, np.load(complete/'checkpoint.npz', allow_pickle=False) as b:
        for key in ('parameters', 'first_moment', 'second_moment', 'losses'):
            np.testing.assert_array_equal(a[key], b[key])
    assert manifest['python'] and len(manifest['source_sha256']) == 64 and 'code_revision' in manifest
    for name in ('checkpoint.npz', 'comparison.png', 'training.png', 'training.csv'):
        assert (resumed/name).stat().st_size > 0
