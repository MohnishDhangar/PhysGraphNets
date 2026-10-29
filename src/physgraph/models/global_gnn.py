"""Global anchor graph on a unit square with affine prescribed boundary data.

Physical nodes send area-weighted messages to fixed coarse anchors. All anchors
and boundary inputs are available to every decoded node. No PDE solution or
manufactured-case parameters enter the forward pass.
"""
import torch
from torch import nn
from physgraph.models.gnn import mlp


class GlobalGraphNetwork(nn.Module):
    def __init__(self,width=32,depth=3):
        super().__init__()
        if type(width) is not int or width<1 or type(depth) is not int or depth<1:
            raise ValueError('positive integer width/depth required')
        axis=torch.linspace(0.,1.,3,dtype=torch.float64)
        yy,xx=torch.meshgrid(axis,axis,indexing='ij')
        self.register_buffer('anchors',torch.stack((xx.ravel(),yy.ravel()),dim=1))
        self.encoder=mlp(5,width,8)
        self.context=mlp(9*8+3+4+1,width,width)
        layers=[nn.Linear(width+2,width),nn.Tanh()]
        for _ in range(depth-1):
            layers.extend((nn.Linear(width,width),nn.Tanh()))
        layers.append(nn.Linear(width,1))
        self.decoder=nn.Sequential(*layers)
        self.double().cpu()

    def forward(self,nodes,edges,edge_index,prescribed,values,graph_index=None):
        if nodes.dtype!=torch.float64 or nodes.device.type!='cpu' or graph_index is None:
            raise ValueError('require CPU float64 inputs and graph membership')
        encoded=self.encoder(nodes[:,:5])
        result=torch.zeros_like(values)
        for group in torch.unique(graph_index):
            selected=graph_index==group
            xy=nodes[selected,:2];mask=prescribed[selected];g=values[selected]
            if torch.any(xy<0) or torch.any(xy>1):
                raise ValueError('global square model requires unit-square coordinates')
            left=torch.isclose(xy[:,0],torch.zeros_like(xy[:,0]))
            right=torch.isclose(xy[:,0],torch.ones_like(xy[:,0]))
            horizontal=torch.isclose(xy[:,1],torch.zeros_like(xy[:,1]))|torch.isclose(xy[:,1],torch.ones_like(xy[:,1]))
            if not torch.all(mask[left|horizontal]):
                raise ValueError('require Dirichlet left/top/bottom square edges')
            full_dirichlet=bool(torch.all(mask[right]))
            if not torch.equal(mask,left|horizontal|(right if full_dirichlet else torch.zeros_like(right))):
                raise ValueError('unsupported square boundary partition')
            design=torch.cat((torch.ones((len(xy),1),dtype=xy.dtype),xy),dim=1)
            bd=design[mask]
            coefficients=torch.linalg.solve(bd.T@bd,bd.T@g[mask])
            if not torch.allclose(bd@coefficients,g[mask],atol=1e-10,rtol=1e-10):
                raise ValueError('global square model requires affine Dirichlet data')
            lifting=design@coefficients
            weight=torch.exp(-((xy[:,None]-self.anchors)**2).sum(2)/(2*.3**2))*nodes[selected,8,None]
            pooled=(weight.T@encoded[selected])/weight.sum(0)[:,None]
            arcs=(graph_index[edge_index[0]]==group)&(edges[:,6]>0.5)
            if torch.any(arcs):
                midpoint=(nodes[edge_index[0,arcs],:2]+nodes[edge_index[1,arcs],:2])/2
                y=midpoint[:,1];flux=edges[arcs,5];length=edges[arcs,2]
                moments=torch.stack([torch.sum(length*flux*y**k)/length.sum() for k in range(4)])
            else:
                moments=torch.zeros(4,dtype=xy.dtype)
            context=self.context(torch.cat((pooled.ravel(),coefficients,moments,
                                             torch.tensor([float(full_dirichlet)],dtype=xy.dtype))))
            correction=self.decoder(torch.cat((xy,context.expand(len(xy),-1)),dim=1)).squeeze(1)
            # Smooth physical lifting; no hard clamping jump at the first free row.
            x,y=xy[:,0],xy[:,1]
            envelope=16*x*(1-x)*y*(1-y) if full_dirichlet else 4*x*y*(1-y)
            predicted=lifting+envelope*correction
            result[selected]=torch.where(mask,g,predicted)
        return result
