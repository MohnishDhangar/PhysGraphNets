import numpy as np
import pytest
torch = pytest.importorskip('torch')
from physgraph.models.gnn import GraphNetwork


def test_equivariance_boundaries_gradients_and_disjoint_graphs():
    torch.manual_seed(3)
    model = GraphNetwork(8, 2)
    x = torch.randn(5, 9, dtype=torch.float64)
    edge = torch.tensor([[0,1,1,2,3,4], [1,0,2,1,4,3]])
    features = torch.randn(6,8,dtype=torch.float64)
    mask = torch.tensor([True,False,False,True,False])
    values = torch.tensor([2.,0.,0.,3.,0.],dtype=torch.float64)
    predicted = model(x,features,edge,mask,values)
    assert torch.equal(predicted[mask],values[mask])
    perm = torch.tensor([4,2,0,3,1])
    inverse = torch.argsort(perm)
    renamed = model(x[perm],features,inverse[edge],mask[perm],values[perm])
    torch.testing.assert_close(renamed,predicted[perm])
    changed=x.clone(); changed[3:] += 10
    torch.testing.assert_close(model(changed,features,edge,mask,values)[:3],predicted[:3])
    predicted.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_trainer_exact_epoch_resume(tmp_path):
    from physgraph.datasets.generate import generate_dataset
    from physgraph.datasets.loading import GraphDataset
    from physgraph.datasets.normalization import fit_normalizer
    from physgraph.training.gnn import train
    generate_dataset(family='parametric_square', instances=[{'physics':{'offset':1+i/10}} for i in range(24)], resolutions=[2,3],output=tmp_path/'data')
    dataset=tmp_path/'data/dataset_manifest.json'
    normal=tmp_path/'normal.json'
    fit_normalizer(GraphDataset(dataset),resolutions=[2]).save(normal)
    args=dict(dataset=dataset,normalization=normal,width=4,depth=1,batch_size=8,seed=3)
    train(**args,epochs=2,output=tmp_path/'full')
    train(**args,epochs=1,output=tmp_path/'part')
    train(**args,epochs=2,output=tmp_path/'resumed',resume=tmp_path/'part/checkpoint.pt')
    a=torch.load(tmp_path/'full/checkpoint.pt',weights_only=True)
    b=torch.load(tmp_path/'resumed/checkpoint.pt',weights_only=True)
    assert a['history']==b['history']
    for key in a['model']:
        assert torch.equal(a['model'][key],b['model'][key])
    for key in a['optimizer']['state']:
        for field in a['optimizer']['state'][key]:
            assert torch.equal(a['optimizer']['state'][key][field],b['optimizer']['state'][key][field])


def test_qualification_aggregation_baseline_and_failure_accounting():
    from types import SimpleNamespace
    from physgraph.experiments.gnn_study import boundary_affine, summarize, assess
    nodes=np.array([[0.,0.],[1.,0.],[0.,1.],[.3,.4]])
    prescribed=np.array([True,True,True,False])
    sample=SimpleNamespace(graph=SimpleNamespace(nodes=nodes),inputs={
        'dirichlet_mask':prescribed,'dirichlet_values':np.array([1.,3.,4.,0.])})
    np.testing.assert_allclose(boundary_affine(sample),1+2*nodes[:,0]+3*nodes[:,1])
    summary=summarize([{'relative_fem_l2':.02,'relative_fem_h1_seminorm':.1},
                       {'relative_fem_l2':.08,'relative_fem_h1_seminorm':.3}])
    assert summary['relative_fem_l2']['mean']==pytest.approx(.05)
    baseline={'relative_fem_l2':{'mean':.2},'relative_fem_h1_seminorm':{'mean':.8}}
    assert not assess(summary,baseline,{'relative_fem_l2':{'max':.1}})
    assert assess(summary,baseline,{'relative_fem_l2':{'max':.01}})


def test_training_cache_excludes_reserved_resolutions_and_test_targets():
    from physgraph.training.gnn import _CachedTrainingData
    from types import SimpleNamespace
    rows=[{'split':s,'resolution':n} for s in ('train','validation','test') for n in (8,16)]
    def load(index):
        assert rows[index]['split']!='test' and rows[index]['resolution']==8
        return index
    cache=_CachedTrainingData(SimpleNamespace(rows=rows,manifest={},load=load),[8])
    assert set(cache.samples)=={0,2}
    with pytest.raises(KeyError):
        cache.load(4)
