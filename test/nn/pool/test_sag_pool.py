from __future__ import division

import torch
from torch_geometric.nn.pool import SAGPooling


def test_sag_pooling():
    in_channels = 16
    edge_index = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
                               [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2]])
    num_nodes = edge_index.max().item() + 1
    x = torch.randn((num_nodes, in_channels))

    pool = SAGPooling(in_channels, ratio=0.5, gnn='GCN')
    assert pool.__repr__() == 'SAGPooling(GCN, 16, ratio=0.5)'

    out = pool(x, edge_index)
    assert out[0].size() == (num_nodes // 2, in_channels)
    assert out[1].size() == (2, 2)

    pool = SAGPooling(in_channels, ratio=0.5, gnn='GAT')
    out = pool(x, edge_index)
    assert out[0].size() == (num_nodes // 2, in_channels)
    assert out[1].size() == (2, 2)

    pool = SAGPooling(in_channels, ratio=0.5, gnn='SAGE')
    out = pool(x, edge_index)
    assert out[0].size() == (num_nodes // 2, in_channels)
    assert out[1].size() == (2, 2)
