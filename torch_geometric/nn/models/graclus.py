import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.data import Batch
from torch_geometric.nn import (GraphConv, graclus, max_pool, global_mean_pool,
                                JumpingKnowledge)


class Graclus(torch.nn.Module):
    r"""A model using the :class:`torch_geometric.nn.Graclus` module from
    the `"Weighted Graph Cuts without
    Eigenvectors: A Multilevel Approach" <http://www.cs.utexas.edu/users/
    inderjit/public_papers/multilevel_pami.pdf>`_ paper.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Hidden node feature dimensionality.
        num_layers (int): Number of GraphConv layers.
    """

    def __init__(self, in_channels, hidden_channels, num_layers):
        super(Graclus, self).__init__()
        self.conv1 = GraphConv(in_channels, hidden_channels, aggr='mean')
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GraphConv(hidden_channels, hidden_channels, aggr='mean'))
        self.jump = JumpingKnowledge(mode='cat')

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.jump.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        xs = [global_mean_pool(x, batch)]
        for i, conv in enumerate(self.convs):
            x = F.relu(conv(x, edge_index))
            xs += [global_mean_pool(x, batch)]
            if i % 2 == 0 and i < len(self.convs) - 1:
                cluster = graclus(edge_index, num_nodes=x.size(0))
                data = Batch(x=x, edge_index=edge_index, batch=batch)
                data = max_pool(cluster, data)
                x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.jump(xs)
        return x

    def __repr__(self):
        return self.__class__.__name__