import torch
from torch_geometric.nn import HeteConv, GCNConv, SAGEConv


def test_hete():
    node_types = {'A': {'x': torch.randn(3, 4)}, 'B': {'x': torch.randn(2, 4)}}
    edge_types = {
        ('A', None, 'A'): {
            'edge_index': torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        },
        ('A', None, 'B'): {
            'edge_index': torch.tensor([[0, 1, 2], [0, 1, 0]]),
            'edge_weight': torch.tensor([0.6, 0.5, 0.7]),
        },
        ('B', None, 'B'): {
            'edge_index': torch.tensor([[0, 1], [1, 0]]),
        },
    }

    convs = {
        ('A', None, 'A'): GCNConv(4, 8),
        ('A', None, 'B'): SAGEConv(4, 8),
        ('B', None, 'B'): GCNConv(4, 8),
    }

    conv = HeteConv(convs, aggr='add')
    conv.reset_parameters()
    out = conv(node_types, edge_types)
    assert len(out.keys()) == 2
    assert out['A'].size() == (3, 8)
    assert out['B'].size() == (2, 8)

    conv.aggr = 'concat'
    out = conv(node_types, edge_types)
    assert len(out.keys()) == 2
    assert out['A'].size() == (3, 8)
    assert out['B'].size() == (2, 16)
