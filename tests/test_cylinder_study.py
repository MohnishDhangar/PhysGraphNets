"""Independent force histories and qualification failure accounting."""
import copy
import numpy as np
import pytest

from physgraph.evaluation.cylinder_flow import published_comparison
from physgraph.evaluation.cylinder_study import compare_cylinder_histories, refinement_acceptance
from physgraph.experiments.cylinder_study import run_cylinder_study
from physgraph.physics.cylinder_flow import CylinderFlowCase


def history(times):
    return {'rows': [dict(time=t, drag=2*t, lift=t, pressure_difference=-t) for t in times],
            'max_drag': 2., 'max_lift': 1., 'final_pressure_difference': -1.,
            'time_max_drag': 1., 'time_max_lift': 1.}


def test_reference_preserves_signed_raw_pressure():
    report = published_comparison({}, CylinderFlowCase(), 0.)
    front, rear = -.43873062102, -.32718923230
    assert report['reference']['final_pressure_difference'] == pytest.approx(front-rear, abs=1e-12)
    assert report['pressure_reference_time'] < 8.
    assert len(report['pressure_archive_sha256']) == 64
    reference = report['reference']
    assert published_comparison(reference, CylinderFlowCase(), 8.)['status'] == 'passed'
    assert published_comparison({**reference, 'final_pressure_difference': -reference['final_pressure_difference']},
                                CylinderFlowCase(), 8.)['status'] == 'failed_acceptance'


def test_history_comparison_known_linear_samples_and_offset():
    coarse = history(np.linspace(.1, 1, 10))
    fine = history(np.linspace(.05, 1, 20))
    exact = compare_cylinder_histories(coarse, fine)
    assert max(exact['normalized_history_rms'].values()) < 1e-14
    shifted = copy.deepcopy(fine)
    for row in shifted['rows']:
        row['lift'] += .1
    comparison = compare_cylinder_histories(coarse, shifted)
    assert comparison['normalized_history_rms']['lift'] == pytest.approx(.1/.47748781595)
    assert not refinement_acceptance(comparison, dict(summary=.03, peak_times=.05, history_rms=.05))['history_rms']


@pytest.mark.parametrize('times', [[.2, .1, 1.], [.2, .5, .9], [.2, np.nan, 1.]])
def test_history_comparison_refuses_invalid_time_grids(times):
    with pytest.raises(ValueError):
        compare_cylinder_histories(history([.1, .5, 1.]), history(times))


@pytest.mark.parametrize('kwargs', [dict(levels=(2, 4)), dict(levels=(2, 4, 7)),
    dict(time_steps=(.02, .01)), dict(time_steps=(.03, .015, .0075)),
    dict(surface_reaction_tolerance=0), dict(refinement_criteria={'summary': .03})])
def test_study_validates_protocol_before_creating_output(tmp_path, kwargs):
    output = tmp_path/'study'
    with pytest.raises(ValueError):
        run_cylinder_study(output=output, **kwargs)
    assert not output.exists()


@pytest.mark.parametrize('external,discrete,expected', [
    (False, True, 'failed_acceptance'), (True, True, 'qualified'),
    (True, False, 'failed_acceptance')])
def test_study_requires_all_checks_and_independent_trajectory(tmp_path, monkeypatch, external, discrete, expected):
    def fake_run(**kwargs):
        h = history(np.linspace(kwargs['dt'], 8., round(8/kwargs['dt'])))
        for row in h['rows']:
            row['surface_drag'], row['surface_lift'] = row['drag'], row['lift']
        return {'metrics': {'history': h, 'status': 'benchmark_candidate_passed',
            'checks': {'residual': discrete}, 'mesh_quality': {}, 'fenicsx': {'passed': True},
            'published': {'status': 'passed', 'reference': {'max_drag': 3., 'max_lift': .5}}}}
    monkeypatch.setattr('physgraph.experiments.cylinder_study.run_cylinder_flow', fake_run)
    report = run_cylinder_study(output=tmp_path/'study', levels=(2, 4, 8),
                                time_steps=(.04, .02, .01), external=external)
    assert len(report['metrics']['runs']) == 5
    assert report['metrics']['checks']['spatial']
    assert report['metrics']['checks']['temporal']
    assert report['metrics']['checks']['independent'] == external
    assert report['metrics']['checks']['discrete'] == discrete
    assert report['metrics']['status'] == expected


def test_runner_exports_short_trajectory_without_claiming_benchmark(tmp_path):
    from physgraph.experiments.cylinder_flow import run_cylinder_flow
    report = run_cylinder_flow(output=tmp_path/'run', level=2, dt=.04, final_time=.08)
    assert report['metrics']['status'] == 'passed_discrete_checks'
    assert report['metrics']['published']['status'] == 'not_assessed'
    with np.load(tmp_path/'run'/'trajectory.npz', allow_pickle=False) as archive:
        assert archive['times'] == pytest.approx([0., .04, .08])
        assert len(archive['pressure']) == 2
    assert (tmp_path/'run'/'forces.csv').read_text().count('\n') == 3
