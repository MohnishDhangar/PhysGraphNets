"""CPU float64 message passing with exact prescribed boundary values."""
import torch
from torch import nn


def mlp(inputs, width, outputs):
    return nn.Sequential(nn.Linear(inputs, width), nn.Tanh(), nn.Linear(width, outputs))


class GraphNetwork(nn.Module):
    def __init__(self, width=32, depth=3):
        super().__init__()
        if type(width) is not int or width < 1 or type(depth) is not int or depth < 1:
            raise ValueError('width and depth must be positive integers')
        self.encoder = mlp(9, width, width)
        self.messages = nn.ModuleList(mlp(2*width+8, width, width) for _ in range(depth))
        self.updates = nn.ModuleList(mlp(2*width, width, width) for _ in range(depth))
        self.decoder = mlp(width, width, 1)
        self.double().cpu()

    def forward(self, nodes, edges, edge_index, prescribed, values, graph_index=None):
        if nodes.device.type != 'cpu' or nodes.dtype != torch.float64:
            raise ValueError('CPU float64 inputs required')
        hidden = self.encoder(nodes)
        source, target = edge_index
        degree = torch.bincount(target, minlength=len(nodes)).clamp_min(1)[:, None]
        for message, update in zip(self.messages, self.updates):
            messages = message(torch.cat((hidden[source], hidden[target], edges), dim=1))
            aggregate = torch.zeros_like(hidden).index_add(0, target, messages)/degree
            hidden = hidden+update(torch.cat((hidden, aggregate), dim=1))
        return torch.where(prescribed, values, self.decoder(hidden).squeeze(1))


def batch_tensors(batch):
    # Copy read-only arrays; targets are deliberately absent from this interface.
    return (torch.tensor(batch.node_features), torch.tensor(batch.edge_features),
            torch.tensor(batch.edge_index), torch.tensor(batch.dirichlet_mask),
            torch.tensor(batch.dirichlet_values), torch.tensor(batch.graph_index))


def make_model(width,depth,architecture='local'):
    if architecture=='local':
        return GraphNetwork(width,depth)
    if architecture=='global_square':
        from physgraph.models.global_gnn import GlobalGraphNetwork
        return GlobalGraphNetwork(width,depth)
    raise ValueError('unknown graph architecture')
