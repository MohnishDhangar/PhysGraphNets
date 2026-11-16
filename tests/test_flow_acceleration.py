"""Profile real constrained systems and reject corrupted scientific fixtures."""
import json
import numpy as np
import pytest
from scipy.sparse import load_npz
from physgraph.experiments.cylinder_flow import run_cylinder_flow
from physgraph.profiling.flow_linear import array_frames, prepare_systems
from physgraph.profiling.flow_linear_worker import trial
from physgraph.experiments.flow_acceleration import run_linear_study


def test_selected_frames_preserve_shape_order_and_values(tmp_path):
    values=np.arange(5*4*2,dtype=float).reshape(5,4,2)
    path=tmp_path/'frames.npz'
    np.savez_compressed(path,velocity=values)
    selected=array_frames(path,'velocity',[4,0,3,4])
    assert set(selected)=={0,3,4}
    for index,value in selected.items():
        np.testing.assert_array_equal(value,values[index])
    with pytest.raises(ValueError,match='index'):
        array_frames(path,'velocity',[5])


def test_archived_newton_fixtures_and_isolated_trials(tmp_path):
    run_cylinder_flow(output=tmp_path/'flow',level=2,dt=.04,final_time=.12)
    report=prepare_systems(tmp_path/'flow/trajectory.npz',tmp_path/'fixtures',steps=[1,3])
    assert len(report['metrics']['cases'])==2
    for case in report['metrics']['cases']:
        folder=tmp_path/'fixtures'/case['path']
        a=load_npz(folder/'matrix.npz')
        with np.load(folder/'vectors.npz') as values:
            assert np.linalg.norm(a@values['reference']-values['rhs'])<1e-10
        result=trial(dict(fixture=str(folder),backend='scipy'))
        assert result['passed'] and result['seconds']>0
    study=run_linear_study(fixtures=tmp_path/'fixtures',output=tmp_path/'timings',
                           backends=['scipy'],threads=[1],repeats=2)
    assert study['metrics']['status']=='completed'
    assert len(study['metrics']['runs'])==4
    assert all(row['passed'] for row in study['metrics']['summary'])
    path=tmp_path/'fixtures/step00001/vectors.npz'
    with path.open('ab') as stream: stream.write(b'changed')
    with pytest.raises(ValueError,match='checksum'):
        run_linear_study(fixtures=tmp_path/'fixtures',output=tmp_path/'bad',
                         backends=['scipy'],threads=[1],repeats=2)
