import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import (GraphConv, EdgePooling, global_mean_pool,
                                JumpingKnowledge)


class EdgePool(torch.nn.Module):
    r"""Model from the `"Towards Graph Pooling by Edge
    Contraction" <https://graphreason.github.io/papers/17.pdf>`_ and
    `"Edge Contraction Pooling for Graph Neural Networks"
    <https://arxiv.org/abs/1905.10990>`_ papers, using the 
    :class:`torch_geometric.nn.EdgePooling` as an edge pooling operator.


    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Hidden node feature dimensionality.
        num_layers (int): Number of GNN layers.
    """
    def __init__(self, in_channels, hidden_channels, num_layers):
        super(EdgePool, self).__init__()
        self.conv1 = GraphConv(in_channels, hidden_channels, aggr='mean')
        self.convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.convs.extend([
            GraphConv(hidden_channels, hidden_channels, aggr='mean')
            for i in range(num_layers - 1)
        ])
        self.pools.extend(
            [EdgePooling(hidden_channels) for i in range((num_layers) // 2)])
        self.jump = JumpingKnowledge(mode='cat')

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        xs = [global_mean_pool(x, batch)]
        for i, conv in enumerate(self.convs):
            x = F.relu(conv(x, edge_index))
            xs += [global_mean_pool(x, batch)]
            if i % 2 == 0 and i < len(self.convs) - 1:
                pool = self.pools[i // 2]
                x, edge_index, batch, _ = pool(x, edge_index, batch=batch)
        x = self.jump(xs)
        return x

    def __repr__(self):
        return self.__class__.__name__