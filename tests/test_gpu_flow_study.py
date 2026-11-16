"""Seeds reproduce archived BDF2 history; failed/slow candidates cannot pass."""
from copy import deepcopy
import numpy as np
import pytest
from physgraph.evaluation.gpu_flow_study import assess_gpu_study
from physgraph.experiments.cylinder_flow import run_cylinder_flow
from physgraph.profiling.flow_segments import prepare_flow_segments
from physgraph.experiments.gpu_flow_segment import run_flow_segment


def test_startup_and_wake_seeds_reproduce_cpu_archive(tmp_path):
    run_cylinder_flow(output=tmp_path/'original',level=2,dt=.02,final_time=.12)
    prepared=prepare_flow_segments(trajectory=tmp_path/'original/trajectory.npz',
                                  output=tmp_path/'seeds',starts=[0,3],steps=2)
    assert len(prepared['metrics']['cases'])==2
    for row in prepared['metrics']['cases']:
        report=run_flow_segment(seed=tmp_path/'seeds'/row['path'],output=tmp_path/row['case'],backend='scipy')
        assert report['metrics']['status']=='passed'
        assert max(report['metrics']['archive_reference_errors'].values())<1e-12
        with np.load(tmp_path/row['case']/'trajectory.npz') as archive:
            np.testing.assert_array_equal(archive['times'],.02*np.arange(row['start_step'],row['start_step']+3))
    with pytest.raises(ValueError,match='steps'):
        prepare_flow_segments(trajectory=tmp_path/'original/trajectory.npz',output=tmp_path/'bad',starts=[5],steps=2)


def test_gate_requires_every_repeat_both_cases_accuracy_and_total_speed():
    runs=[dict(case=case,backend=backend,repeat=r,status='passed',process_wall_seconds=seconds)
          for case in ('startup','wake') for backend,seconds in (('mumps',3.),('rocm_gmres',1.)) for r in range(3)]
    comparisons=[dict(case=case,repeat=r,passed=True) for case in ('startup','wake') for r in range(3)]
    def assess(data=runs,checks=comparisons):
        return assess_gpu_study(data,checks,['startup','wake'],repeats=3)
    assert assess()['adopt']
    assert not assess(runs[:-1])['adopt']
    bad=deepcopy(runs);bad[-1]['status']='failed'
    assert not assess(bad)['numerically_valid']
    duplicate=deepcopy(runs);duplicate[-1]['repeat']=0
    assert not assess(duplicate)['numerically_valid']
    slow=deepcopy(runs)
    for row in slow:
        if row['backend']=='rocm_gmres':row['process_wall_seconds']=4.
    result=assess(slow)
    assert result['numerically_valid'] and not result['adopt']
    bad_checks=deepcopy(comparisons);bad_checks[-1]['passed']=False
    assert not assess(checks=bad_checks)['adopt']
