import pytest
from physgraph.experiments.pinn_cost import select_candidate, run_cost_study
from physgraph.profiling.cost_worker import trial


def test_selection_requires_all_repeats_and_both_accuracy_targets():
    target={'relative_l2':.01,'relative_h1_seminorm':.05}
    rows=[]
    for candidate,times,errors in [('a',[1,2],[.02,.005]),('b',[3,4],[.005,.005])]:
        for t,e in zip(times,errors):
            rows.append({'candidate':candidate,'status':'completed','result':{'total_seconds':t,
                'errors':{'relative_l2':e,'relative_h1_seminorm':.04}}})
    assert select_candidate(rows,target)=={'candidate':'b','median_seconds':3.5}
    rows[-1]={'candidate':'b','status':'failed'}
    assert select_candidate(rows,target) is None


def test_fem_worker_reports_accuracy_and_memory():
    r=trial({'solver':'fem','case':'affine','resolution':4})
    assert r['errors']['relative_l2']<1e-12
    assert r['mesh']['nodes']==25
    assert r['total_seconds']>0 and r['peak_rss_bytes_before_evaluation']>0


def test_cost_config_validates_before_output(tmp_path):
    with pytest.raises(ValueError,match='two'):
        run_cost_study(cases=['sine'],resolutions=[4],step_counts=[1],repeats=1,
            target={'relative_l2':.01,'relative_h1_seminorm':.1},output=tmp_path/'out')
    assert not (tmp_path/'out').exists()
