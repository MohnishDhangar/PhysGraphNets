"""Exact differentiable P1 field/gradient errors, averaged per physical graph."""
import torch


def p1_errors(prediction, batch):
    triangles=torch.tensor(batch.triangles)
    points=torch.tensor(batch.node_features[:,:2])[triangles]
    a,b,c=points[:,0],points[:,1],points[:,2]
    twice=(b[:,0]-a[:,0])*(c[:,1]-a[:,1])-(b[:,1]-a[:,1])*(c[:,0]-a[:,0])
    if torch.any(twice<=0):
        raise ValueError('P1 loss requires positively oriented triangles')
    gradients=torch.stack((torch.stack((b[:,1]-c[:,1],c[:,0]-b[:,0]),dim=1),
                           torch.stack((c[:,1]-a[:,1],a[:,0]-c[:,0]),dim=1),
                           torch.stack((a[:,1]-b[:,1],b[:,0]-a[:,0]),dim=1)),dim=1)/twice[:,None,None]
    target=torch.tensor(batch.fem_solution)[triangles]
    error=prediction[triangles]-target
    area=twice/2
    def integrals(values):
        l2=area/12*(values.square().sum(1)+values.sum(1).square())
        h1=area*(values[:,:,None]*gradients).sum(1).square().sum(1)
        return l2,h1
    e2,eh=integrals(error);t2,th=integrals(target)
    results=[]
    for lo,hi in zip(batch.triangle_ptr[:-1],batch.triangle_ptr[1:]):
        # Floor only protects zero-reference norms; matches evaluation contract.
        results.append(torch.stack((e2[lo:hi].sum()/t2[lo:hi].sum().clamp_min(1e-24),
                                    eh[lo:hi].sum()/th[lo:hi].sum().clamp_min(1e-24))))
    return torch.stack(results)


def sobolev_loss(prediction,batch,h1_weight=1.):
    if h1_weight<0:
        raise ValueError('H1 weight must be nonnegative')
    errors=p1_errors(prediction,batch)
    return (errors[:,0]+h1_weight*errors[:,1]).mean()
