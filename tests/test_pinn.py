from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from physgraph.models.pinn import TanhMLP, constrained_field
from physgraph.physics.pinn import get_pinn_case, DIRICHLET_CASE_NAMES as CASE_NAMES
from physgraph.losses.pinn import loss, loss_gradient, strong_residual
from physgraph.training.pinn import TrainingConfig, train, save_checkpoint, load_checkpoint
from physgraph.evaluation.pinn import evaluate_pinn
from physgraph.mesh.structured import unit_square_mesh
from physgraph.io.results import load_field


def test_network_spatial_derivatives_match_independent_differences():
    model = TanhMLP(4, 2)
    parameters = model.initialize(np.random.default_rng(8))
    points = np.array([[.2, .3], [.45, .8], [.7, .1]])
    values, gradient, diagonal = model.derivatives(parameters, points)
    h = 1e-4
    for direction in range(2):
        shift = np.eye(2)[direction]*h
        plus = model.derivatives(parameters, points+shift)[0]
        minus = model.derivatives(parameters, points-shift)[0]
        np.testing.assert_allclose(gradient[:, direction], (plus-minus)/(2*h), atol=1e-7, rtol=1e-6)
        np.testing.assert_allclose(diagonal[:, direction], (plus-2*values+minus)/h**2, atol=2e-7, rtol=1e-5)


@pytest.mark.parametrize('name', CASE_NAMES)
def test_lifting_enforces_all_nonzero_boundary_data(name):
    case = get_pinn_case(name)
    model = TanhMLP(4, 1)
    parameters = model.initialize(np.random.default_rng(12))
    t = np.linspace(0, 1, 19)
    points = np.vstack((np.column_stack((t, t*0)), np.column_stack((t, t*0+1)),
                        np.column_stack((t*0, t)), np.column_stack((t*0+1, t))))
    values = constrained_field(model, parameters, points, case.lifting)[0]
    np.testing.assert_allclose(values, case.problem.boundary.dirichlet(points), atol=5e-16, rtol=0)


@pytest.mark.parametrize('name', ['affine', 'affine_diffusion'])
def test_exact_affine_strong_residual(name):
    model = TanhMLP(3, 2)
    parameters = np.zeros(model.parameter_count)
    points = np.random.default_rng(14).random((20, 2))
    np.testing.assert_allclose(strong_residual(model, parameters, points, get_pinn_case(name)), 0., atol=2e-15)


def test_variable_diffusion_strong_form_against_flux_differences():
    model = TanhMLP(3, 2)
    params = model.initialize(np.random.default_rng(13))
    case = get_pinn_case('variable_sine')
    points = np.array([[.2, .3], [.4, .7], [.9, .2]])
    u = constrained_field(model, params, points, case.lifting)[0]
    div_flux = np.zeros(len(points))
    h = 1e-5
    for d in range(2):
        offset = np.eye(2)[d]*h
        plus = constrained_field(model, params, points+offset, case.lifting)[1][:, d]*case.problem.diffusion(points+offset)
        minus = constrained_field(model, params, points-offset, case.lifting)[1][:, d]*case.problem.diffusion(points-offset)
        div_flux += (plus-minus)/(2*h)
    expected = -div_flux+case.problem.reaction(points)*u-case.problem.forcing(points)
    np.testing.assert_allclose(strong_residual(model, params, points, case), expected, rtol=1e-8, atol=1e-7)


def test_complex_step_parameter_gradients_match_central_differences():
    model = TanhMLP(3, 2)
    params = model.initialize(np.random.default_rng(16))
    points = np.random.default_rng(17).random((20, 2))
    case = get_pinn_case('variable_sine')
    gradient = loss_gradient(model, params, points, case)
    numerical = []
    for i in range(len(params)):
        step = np.zeros_like(params); step[i] = 1e-5
        numerical.append((loss(model, params+step, points, case)-loss(model, params-step, points, case))/2e-5)
    np.testing.assert_allclose(gradient, numerical, rtol=2e-7, atol=1e-8)


def test_training_reduces_residual_and_independent_error():
    config = TrainingConfig(width=4, depth=1, collocation=48, learning_rate=.01)
    initial = train(config, 0)
    trained = train(config, 200)
    model, case, mesh = TanhMLP(4, 1), get_pinn_case('sine'), unit_square_mesh(6)
    _, before = evaluate_pinn(mesh, model, initial.parameters, case)
    _, after = evaluate_pinn(mesh, model, trained.parameters, case)
    assert trained.losses[-1] < initial.losses[0]*.1
    assert after['relative_l2'] < before['relative_l2']*.2
    assert after['holdout_residual_rms'] < before['holdout_residual_rms']*.3
    assert after['max_boundary_error'] < 1e-14  # sin(pi) in the NumPy reference is roundoff-sized.


def test_checkpoint_resume_is_identical_to_uninterrupted_training(tmp_path):
    config = TrainingConfig(width=3, depth=1, collocation=16)
    complete = train(config, 30)
    path = tmp_path/'checkpoint.npz'
    first = train(config, 12, checkpoint=path, checkpoint_every=5)
    resumed = train(config, 30, resume=path)
    assert first.step == 12 and resumed.step == 30
    for key in ('parameters', 'first_moment', 'second_moment', 'points', 'losses'):
        np.testing.assert_array_equal(getattr(complete, key), getattr(resumed, key))
    assert complete.rng_state == resumed.rng_state
    with pytest.raises(ValueError, match='configuration'):
        load_checkpoint(path, replace(config, seed=8))
    with pytest.raises(ValueError, match='precede'):
        train(config, 2, resume=path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    arrays['parameters'][0] = np.nan
    np.savez_compressed(tmp_path/'broken.npz', **arrays)
    with pytest.raises(ValueError, match='malformed'):
        load_checkpoint(tmp_path/'broken.npz', config)


def test_direct_neural_quadrature_has_zero_error_for_affine_reference():
    mesh, model = unit_square_mesh(4), TanhMLP(3, 1)
    _, errors = evaluate_pinn(mesh, model, np.zeros(model.parameter_count), get_pinn_case('affine'))
    assert errors['absolute_l2'] < 1e-14
    assert errors['absolute_h1_seminorm'] < 1e-14
    assert errors['holdout_residual_rms'] == 0


def test_cli_preserves_fem_and_prediction_as_separate_fields(tmp_path):
    config = tmp_path/'config.json'
    config.write_text(json.dumps({'case':'affine', 'width':3, 'depth':1, 'collocation':16, 'steps':2, 'nx':4}))
    output = tmp_path/'run'
    subprocess.run([sys.executable, 'scripts/run_pinn.py', '--config', str(config), '--steps', '3', '--output', str(output)],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, check=True)
    manifest = json.loads((output/'manifest.json').read_text())
    assert manifest['config']['steps'] == 3
    assert manifest['metrics']['status'] == 'completed'
    assert manifest['metrics']['fem']['max_nodal_error'] < 1e-12
    assert manifest['metrics']['pinn']['max_nodal_error'] > 1e-6
    fem, _ = load_field(output/'fem_reference.npz')
    with np.load(output/'comparison.npz', allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive['fem_solution'], fem.solution)
        assert not np.array_equal(archive['fem_solution'], archive['pinn_prediction'])
    assert manifest['python'] and len(manifest['source_sha256']) == 64
    assert 'code_revision' in manifest
    for name in ('checkpoint.npz', 'comparison.png', 'training.png', 'training.csv'):
        assert (output/name).stat().st_size > 0


@pytest.mark.parametrize('kwargs', [{'width':0}, {'depth':True}, {'collocation':1}, {'seed':-1},
    {'learning_rate':np.nan}, {'learning_rate':0}, {'case':'mixed_polynomial'}])
def test_invalid_training_settings(kwargs):
    with pytest.raises(ValueError):
        TrainingConfig(**kwargs)


def test_numpy_backend_remains_usable_without_torch(tmp_path):
    code = '''
import sys
sys.modules['torch'] = None
from physgraph.training.pinn import TrainingConfig, train
from physgraph.experiments.pinn import run_experiment
assert train(TrainingConfig(width=3, depth=1, collocation=16), 1).step == 1
try:
    run_experiment(backend='torch', output=sys.argv[1])
except ValueError as exc:
    assert 'PyTorch backend unavailable' in str(exc)
else:
    raise AssertionError('missing optional dependency must fail clearly')
'''
    output = tmp_path/'missing_backend'
    subprocess.run([sys.executable, '-c', code, str(output)],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, check=True)
    assert not output.exists()
