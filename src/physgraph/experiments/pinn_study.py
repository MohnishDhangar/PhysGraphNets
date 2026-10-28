"""Sequential controlled seed/collocation sweeps with explicit failure accounting."""
from itertools import product
from pathlib import Path
from time import perf_counter
import numpy as np

from physgraph.experiments.pinn import run_experiment
from physgraph.evaluation.acceptance import validate_thresholds
from physgraph.io.manifest import write_manifest
from physgraph.training.pinn import TrainingConfig
from physgraph.evaluation.pinn import validate_evaluation


def summarize(rows):
    complete = [row for row in rows if row['status'] == 'completed']
    statistics = {}
    for key in ('relative_l2', 'relative_h1_seminorm', 'holdout_residual_relative_rms'):
        values = [row['metrics']['pinn'][key] for row in complete]
        statistics[key] = ({'median': float(np.median(values)), 'min': float(np.min(values)),
            'max': float(np.max(values)), 'std': float(np.std(values)), 'count': len(values)} if values else None)
    accepted = sum(row['metrics']['acceptance']['status'] == 'passed' for row in complete)
    return {'attempted': len(rows), 'completed': len(complete), 'execution_failures': len(rows)-len(complete),
            'accepted': accepted, 'acceptance_rate_all_attempts': accepted/len(rows),
            'statistics_completed_only': statistics}


def run_study(*, cases, seeds, collocation_counts, acceptance, output, training=None,
              validation_seed=10007, holdout_points=4096, boundary_points=257, evaluation_quadrature_order=5):
    started = perf_counter()
    training = dict(training or {})
    allowed = {'backend', 'torch_threads', 'width', 'depth', 'learning_rate', 'steps', 'nx', 'checkpoint_every', 'neumann_points', 'neumann_weight'}
    if set(training)-allowed:
        raise ValueError('unsupported study training settings')
    for name, values in (('cases', cases), ('seeds', seeds), ('collocation_counts', collocation_counts)):
        if not isinstance(values, list) or not values or len(set(values)) != len(values):
            raise ValueError(f'{name} must be a nonempty list without duplicates')
    if not isinstance(acceptance, dict) or set(acceptance) != set(cases):
        raise ValueError('declare acceptance thresholds for every case before running the study')
    for case, seed, count in product(cases, seeds, collocation_counts):
        TrainingConfig(case=case, seed=seed, collocation=count,
            **{k: v for k, v in training.items() if k in ('width', 'depth', 'learning_rate', 'neumann_points', 'neumann_weight')})
        validate_thresholds(acceptance[case])
        if acceptance[case] is None:
            raise ValueError('study requires explicit acceptance thresholds')
    validate_evaluation(evaluation_quadrature_order, validation_seed, holdout_points, boundary_points)
    dest = Path(output)
    if dest.exists() and (not dest.is_dir() or any(dest.iterdir())):
        raise ValueError('study output directory must be empty')
    dest.mkdir(parents=True, exist_ok=True)
    config = {'cases': cases, 'seeds': seeds, 'collocation_counts': collocation_counts,
        'acceptance': acceptance, 'training': training, 'validation_seed': validation_seed,
        'holdout_points': holdout_points, 'boundary_points': boundary_points,
        'evaluation_quadrature_order': evaluation_quadrature_order,
        'scope': 'spatial validation on individual manufactured PDEs; no instance generalization claim'}
    rows = []
    def save(status):
        groups = {f'{case}/n{count}': summarize([r for r in rows if r['case'] == case and r['collocation'] == count])
            for case, count in product(cases, collocation_counts)
            if any(r['case'] == case and r['collocation'] == count for r in rows)}
        return write_manifest(dest/'study_manifest.json', config,
            {'status': status, 'runs': rows, 'groups': groups}, perf_counter()-started)
    save('running')
    for case, count, seed in product(cases, collocation_counts, seeds):
        folder = dest/f'{case}_n{count}_seed{seed}'
        row = {'case': case, 'collocation': count, 'seed': seed, 'manifest': str(folder/'manifest.json')}
        try:
            result = run_experiment(**training, case=case, seed=seed, collocation=count, output=folder,
                acceptance=acceptance[case], validation_seed=validation_seed, holdout_points=holdout_points,
                boundary_points=boundary_points, evaluation_quadrature_order=evaluation_quadrature_order)
            row.update(status='completed', metrics=result['metrics'])
        except (ValueError, FloatingPointError, RuntimeError, OSError) as exc:
            row.update(status='failed', reason=str(exc))
        rows.append(row)
        save('running')
    return save('completed_with_failures' if any(r['status'] == 'failed' for r in rows) else 'completed')
