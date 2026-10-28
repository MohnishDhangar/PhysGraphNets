import json
import numpy as np
import pytest
from physgraph.evaluation.acceptance import assess
from physgraph.evaluation.pinn import evaluate_pinn
from physgraph.experiments.pinn_study import run_study, summarize
from physgraph.models.pinn import TanhMLP
from physgraph.mesh.structured import unit_square_mesh
from physgraph.physics.pinn import get_pinn_case
from physgraph.training.pinn import TrainingConfig, train, load_checkpoint


def test_checkpoint_rejects_changed_source_and_legacy_schema(tmp_path, monkeypatch):
    path = tmp_path/'state.npz'
    config = TrainingConfig(width=2, depth=1, collocation=8)
    train(config, 1, checkpoint=path)
    import physgraph.training.pinn as module
    monkeypatch.setattr(module, 'training_fingerprint', lambda: 'changed')
    with pytest.raises(ValueError, match='fingerprint'):
        load_checkpoint(path, config)
    with np.load(path) as data:
        arrays = dict(data)
    metadata = json.loads(arrays['metadata'].item())
    metadata['schema_version'] = 1
    arrays['metadata'] = np.array(json.dumps(metadata))
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match='schema'):
        load_checkpoint(path, config)


def test_evaluation_configuration_and_affine_streaming():
    model = TanhMLP(2, 1)
    _, metrics = evaluate_pinn(unit_square_mesh(24), model, np.zeros(model.parameter_count),
        get_pinn_case('affine'), validation_seed=52, holdout_points=37, boundary_points=19)
    assert metrics['holdout_points'] == 37 and metrics['validation_seed'] == 52
    assert metrics['absolute_l2'] < 1e-14 and metrics['absolute_h1_seminorm'] < 1e-14
    with pytest.raises(ValueError, match='holdout_points'):
        evaluate_pinn(unit_square_mesh(2), model, np.zeros(model.parameter_count),
            get_pinn_case('affine'), holdout_points=0)


def test_acceptance_is_separate_and_rejects_invalid_limits():
    assert assess({'relative_l2': .2}, {'relative_l2': .1})['status'] == 'failed'
    assert assess({'relative_l2': .1}, {'relative_l2': .1})['status'] == 'passed'
    assert assess({}, None)['status'] == 'not_assessed'
    for threshold in ({}, {'typo': 1}, {'relative_l2': float('nan')}, {'relative_l2': True}):
        with pytest.raises(ValueError):
            assess({}, threshold)


def test_study_keeps_failures_and_uses_common_evaluation(tmp_path, monkeypatch):
    import physgraph.experiments.pinn_study as module
    calls = []
    def experiment(**kwargs):
        calls.append(kwargs)
        if kwargs['seed'] == 2:
            raise FloatingPointError('injected divergence')
        return {'metrics': {'pinn': {'relative_l2': .2, 'relative_h1_seminorm': .3,
            'holdout_residual_relative_rms': .4}, 'acceptance': {'status': 'failed'}}}
    monkeypatch.setattr(module, 'run_experiment', experiment)
    result = run_study(cases=['sine'], seeds=[1,2], collocation_counts=[8],
        acceptance={'sine': {'relative_l2': .1}}, output=tmp_path/'study')
    group = result['metrics']['groups']['sine/n8']
    assert group['attempted'] == 2 and group['execution_failures'] == 1 and group['accepted'] == 0
    assert group['statistics_completed_only']['relative_l2']['median'] == .2
    assert len({call['validation_seed'] for call in calls}) == 1
    assert result['metrics']['status'] == 'completed_with_failures'
    assert (tmp_path/'study/study_manifest.json').exists()


def test_small_real_study(tmp_path):
    result = run_study(cases=['affine'], seeds=[1,2], collocation_counts=[8],
        acceptance={'affine': {'max_boundary_error': 1e-12}}, output=tmp_path/'study',
        training={'width':2,'depth':1,'steps':1,'nx':2}, holdout_points=20)
    group = result['metrics']['groups']['affine/n8']
    assert group['completed'] == group['accepted'] == 2
    for row in result['metrics']['runs']:
        assert row['metrics']['fem']['max_nodal_error'] < 1e-12
