from __future__ import division

import torch
from torch_geometric.nn import (ASAP_Pooling, GraphConv, GCNConv)


def test_asap_pooling():
    in_channels = 16
    edge_index = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
                               [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2]])
    num_nodes = edge_index.max().item() + 1
    x = torch.randn((num_nodes, in_channels))

    for GNN in [GraphConv, GCNConv]:
        pool = ASAP_Pooling(in_channels, ratio=0.5, GNN=GNN, dropout_att=0,
                            negative_slope=0.2, self_loops=False)
        assert pool.__repr__() == ('ASAP_Pooling({}, 16, ratio=0.5, 0, '
                                   '0.2, False)').format(GNN.__name__)
        out = pool(x, edge_index)
        assert out[0].size() == (num_nodes // 2, in_channels)
        assert out[1].size() == (2, 2)

        pool = ASAP_Pooling(in_channels, ratio=0.5, GNN=GNN, dropout_att=0,
                            negative_slope=0.2, self_loops=True)
        assert pool.__repr__() == ('ASAP_Pooling({}, 16, ratio=0.5, 0, '
                                   '0.2, True)').format(GNN.__name__)
        out = pool(x, edge_index)
        assert out[0].size() == (num_nodes // 2, in_channels)
        assert out[1].size() == (2, 4)
