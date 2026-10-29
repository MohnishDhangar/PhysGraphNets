from dataclasses import replace
import numpy as np
import pytest
torch=pytest.importorskip('torch')
from physgraph.datasets.generate import generate_dataset
from physgraph.datasets.loading import GraphDataset
from physgraph.datasets.normalization import fit_normalizer
from physgraph.graph.batching import batch_graphs
from physgraph.models.gnn import make_model,batch_tensors
from physgraph.losses.gnn import p1_errors,sobolev_loss
from physgraph.evaluation.graph import prediction_metrics


@pytest.fixture(scope='module')
def repair_data(tmp_path_factory):
    path=tmp_path_factory.mktemp('gnn_repairs')/'data'
    generate_dataset(family='parametric_square',instances=[{'physics':{'offset':1+i/20,
        'boundary_mode':'dirichlet' if i%2==0 else 'right_neumann'}} for i in range(16)],
        resolutions=[4,8],output=path)
    data=GraphDataset(path/'dataset_manifest.json')
    return data,fit_normalizer(data,resolutions=[4])


def test_p1_loss_matches_evaluation_and_has_correct_gradients(repair_data):
    data,normal=repair_data
    sample=next(data.samples('train'))
    batch=batch_graphs([sample],normal)
    prediction=torch.tensor(sample.fem_solution+.02*np.sin(4*sample.graph.nodes[:,0]),requires_grad=True)
    errors=p1_errors(prediction,batch).sqrt()[0].detach().numpy()
    metrics=prediction_metrics(sample,prediction.detach().numpy())
    np.testing.assert_allclose(errors,[metrics['relative_fem_l2'],metrics['relative_fem_h1_seminorm']],atol=1e-14)
    assert torch.autograd.gradcheck(lambda p:sobolev_loss(p,batch), (prediction,),eps=1e-6,atol=1e-5)
    assert sobolev_loss(torch.tensor(sample.fem_solution),batch).item()==0
    shifted=p1_errors(torch.tensor(sample.fem_solution+.1),batch)
    assert shifted[0,1].item()<1e-25


def test_global_model_equivariance_and_batch_isolation(repair_data):
    data,normal=repair_data
    samples=list(data.samples('train'))[:2]
    torch.manual_seed(8)
    model=make_model(8,2,'global_square')
    batch=batch_graphs(samples,normal)
    values=model(*batch_tensors(batch))
    start=0
    for sample in samples:
        single=batch_graphs([sample],normal)
        n=len(sample.graph.nodes)
        torch.testing.assert_close(model(*batch_tensors(single)),values[start:start+n])
        start+=n
    args=list(batch_tensors(batch))
    perm=torch.randperm(len(values));inverse=torch.argsort(perm)
    args[0]=args[0][perm];args[2]=inverse[args[2]]
    for i in (3,4,5):args[i]=args[i][perm]
    torch.testing.assert_close(model(*args),values[perm])
    assert torch.equal(values[torch.tensor(batch.dirichlet_mask)],torch.tensor(batch.dirichlet_values)[torch.tensor(batch.dirichlet_mask)])
    values.square().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_global_boundary_shift_reaches_remote_interior(repair_data):
    data,normal=repair_data
    sample=next(s for s in data.samples('train') if s.resolution==8)
    batch=batch_graphs([sample],normal)
    args=list(batch_tensors(batch))
    center=int(np.argmin(np.sum((sample.graph.nodes-.5)**2,axis=1)))
    torch.manual_seed(4)
    global_model=make_model(8,2,'global_square')
    before=global_model(*args)[center].item()
    args[4]=args[4]+args[3].to(torch.float64)
    after=global_model(*args)[center].item()
    assert abs(after-before)>.1


def test_repaired_trainer_exact_resume(repair_data,tmp_path):
    from physgraph.training.gnn import train
    data,normal=repair_data
    normal.save(tmp_path/'normal.json')
    args=dict(dataset=data.path,normalization=tmp_path/'normal.json',width=8,depth=2,
              batch_size=3,seed=6,architecture='global_square',objective='sobolev',decay_every=1)
    train(**args,epochs=3,output=tmp_path/'full')
    train(**args,epochs=1,output=tmp_path/'part')
    train(**args,epochs=3,output=tmp_path/'resumed',resume=tmp_path/'part/checkpoint.pt')
    a=torch.load(tmp_path/'full/checkpoint.pt',weights_only=True)
    b=torch.load(tmp_path/'resumed/checkpoint.pt',weights_only=True)
    assert a['history']==b['history']
    assert a['config']['objective']=='sobolev'
    assert a['history'][-1]['learning_rate']==pytest.approx(.00025)
    for key in a['model']:assert torch.equal(a['model'][key],b['model'][key])
    for key in a['optimizer']['state']:
        for field in a['optimizer']['state'][key]:
            assert torch.equal(a['optimizer']['state'][key][field],b['optimizer']['state'][key][field])
