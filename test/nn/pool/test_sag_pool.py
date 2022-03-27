import torch

from torch_geometric.nn import (
    GATConv,
    GCNConv,
    GraphConv,
    SAGEConv,
    SAGPooling,
)


def test_sag_pooling():
    in_channels = 16
    edge_index = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
                               [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2]])
    num_nodes = edge_index.max().item() + 1
    x = torch.randn((num_nodes, in_channels))

    for GNN in [GraphConv, GCNConv, GATConv, SAGEConv]:
        pool = SAGPooling(in_channels, ratio=0.5, GNN=GNN)
        assert pool.__repr__() == (f'SAGPooling({GNN.__name__}, 16, '
                                   f'ratio=0.5, multiplier=1.0)')
        out = pool(x, edge_index)
        assert out[0].size() == (num_nodes // 2, in_channels)
        assert out[1].size() == (2, 2)

        pool = SAGPooling(in_channels, ratio=None, GNN=GNN, min_score=0.1)
        assert pool.__repr__() == (f'SAGPooling({GNN.__name__}, 16, '
                                   f'min_score=0.1, multiplier=1.0)')
        out = pool(x, edge_index)
        assert out[0].size(0) <= x.size(0) and out[0].size(1) == (16)
        assert out[1].size(0) == 2 and out[1].size(1) <= edge_index.size(1)

        pool = SAGPooling(in_channels, ratio=2, GNN=GNN)
        assert pool.__repr__() == (f'SAGPooling({GNN.__name__}, 16, '
                                   f'ratio=2, multiplier=1.0)')
        out = pool(x, edge_index)
        assert out[0].size() == (2, in_channels)
        assert out[1].size() == (2, 2)
