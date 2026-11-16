"""Predeclared complete cylinder benchmark with separate mesh/time studies."""
from pathlib import Path
from time import perf_counter
import numpy as np

from physgraph.evaluation.cylinder_study import compare_cylinder_histories, refinement_acceptance
from physgraph.experiments.cylinder_flow import run_cylinder_flow
from physgraph.io.manifest import environment_manifest, write_manifest
from physgraph.fem.flow_linear_solver import backend_identity


def run_cylinder_study(*, output, levels=(4, 8, 16), time_steps=(.02, .01, .005),
                       external=True, refinement_criteria=None, surface_reaction_tolerance=.05,
                       linear_backend='scipy'):
    levels, time_steps = list(levels), list(time_steps)
    if (len(levels) < 3 or any(type(n) is not int or n < 2 for n in levels)
            or any(b != 2*a for a, b in zip(levels, levels[1:]))):
        raise ValueError('levels require at least three successively doubled integers >=2')
    if (len(time_steps) < 3 or not np.isfinite(time_steps).all()
            or any(dt <= 0 or not np.isclose(8/dt, round(8/dt), rtol=0, atol=1e-9)
                   for dt in time_steps)
            or any(not np.isclose(a, 2*b, rtol=0, atol=1e-12)
                   for a, b in zip(time_steps, time_steps[1:]))):
        raise ValueError('time_steps require at least three successively halved steps dividing 8')
    criteria = dict(summary=.03, peak_times=.05, history_rms=.05)
    if refinement_criteria is not None:
        if set(refinement_criteria) != set(criteria):
            raise ValueError('refinement criteria require summary, peak_times, history_rms')
        criteria = dict(refinement_criteria)
    if (not np.isfinite([*criteria.values(), surface_reaction_tolerance]).all()
            or min(*criteria.values(), surface_reaction_tolerance) <= 0):
        raise ValueError('acceptance tolerances must be positive and finite')
    dest = Path(output)
    if dest.exists() and (not dest.is_dir() or any(dest.iterdir())):
        raise ValueError('study output must be empty; preserve prior attempts')
    dest.mkdir(parents=True, exist_ok=True)
    config = dict(levels=levels, time_steps=time_steps, external=bool(external), final_time=8.,
        linear_backend=backend_identity(linear_backend),
        refinement_criteria=criteria, surface_reaction_tolerance=surface_reaction_tolerance,
        spatial_dt=time_steps[-1], temporal_level=levels[-1],
        external_policy='every step of finest mesh/finest timestep trajectory',
        domain_policy='published fixed domain; changed length is a separate sensitivity case',
        acceptance_policy='all discrete checks; finest benchmark; finest mesh/time pairs; independent trajectory')
    started = perf_counter()
    identity = environment_manifest()
    metrics = {'status': 'running', 'runs': [], 'spatial': [], 'temporal': []}

    def persist():
        # Detect source changes during a long scientific study instead of relabeling it.
        if environment_manifest()['source_sha256'] != identity['source_sha256']:
            raise RuntimeError('source changed during cylinder study; start a fresh study')
        return write_manifest(dest/'manifest.json', config, metrics, perf_counter()-started)

    persist()
    cache = {}

    def execute(level, dt):
        key = (level, dt)
        if key not in cache:
            label = f'level{level}_dt{dt:g}'
            report = run_cylinder_flow(output=dest/label, level=level, dt=dt,
                external=external and level == levels[-1] and dt == time_steps[-1],
                linear_backend=linear_backend)
            cache[key] = report['metrics']
            metrics['runs'].append({'path': label+'/manifest.json', 'level': level, 'dt': dt,
                'status': report['metrics']['status'], 'checks': report['metrics']['checks'],
                'published': report['metrics']['published'], 'mesh': report['metrics']['mesh_quality']})
            persist()
        return cache[key]

    try:
        # At fixed finest dt vary only space; at fixed finest space vary only dt.
        spatial = [execute(n, time_steps[-1]) for n in levels]
        temporal = [execute(levels[-1], dt) for dt in time_steps]
        for name, runs in (('spatial', spatial), ('temporal', temporal)):
            for a, b in zip(runs, runs[1:]):
                comparison = compare_cylinder_histories(a['history'], b['history'])
                metrics[name].append({**comparison, 'checks': refinement_acceptance(comparison, criteria)})
        finest = cache[(levels[-1], time_steps[-1])]
        history = finest['history']
        reference = finest['published']['reference']
        discrepancy = {
            key: float(max(abs(r['surface_'+key]-r[key]) for r in history['rows'])/reference['max_'+key])
            for key in ('drag', 'lift')}
        metrics['surface_reaction_max_difference'] = discrepancy
        checks = dict(
            discrete=all(all(r['checks'].values()) for r in metrics['runs']),
            benchmark=finest['published']['status'] == 'passed',
            spatial=all(metrics['spatial'][-1]['checks'].values()),
            temporal=all(metrics['temporal'][-1]['checks'].values()),
            surface_reaction=max(discrepancy.values()) <= surface_reaction_tolerance,
            independent=bool(external and finest.get('fenicsx', {}).get('passed', False)),
        )
        metrics['checks'] = checks
        metrics['status'] = 'qualified' if all(checks.values()) else 'failed_acceptance'
        return persist()
    except Exception as exc:
        metrics.update(status='failed', reason=f'{type(exc).__name__}: {exc}')
        # Preserve launch identity explicitly if current source has changed.
        metrics['launch_source_sha256'] = identity['source_sha256']
        write_manifest(dest/'manifest.json', config, metrics, perf_counter()-started)
        raise
