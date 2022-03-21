import pytest
import torch

from torch_geometric.data import HeteroData
from torch_geometric.nn import (
    GATConv,
    GCNConv,
    HeteroConv,
    Linear,
    MessagePassing,
    SAGEConv,
)


def get_edge_index(num_src_nodes, num_dst_nodes, num_edges):
    row = torch.randint(num_src_nodes, (num_edges, ), dtype=torch.long)
    col = torch.randint(num_dst_nodes, (num_edges, ), dtype=torch.long)
    return torch.stack([row, col], dim=0)


@pytest.mark.parametrize('aggr', ['sum', 'mean', 'min', 'max', None])
def test_hetero_conv(aggr):
    data = HeteroData()
    data['paper'].x = torch.randn(50, 32)
    data['author'].x = torch.randn(30, 64)
    data['paper', 'paper'].edge_index = get_edge_index(50, 50, 200)
    data['paper', 'author'].edge_index = get_edge_index(50, 30, 100)
    data['author', 'paper'].edge_index = get_edge_index(30, 50, 100)
    data['paper', 'paper'].edge_weight = torch.rand(200)

    conv = HeteroConv(
        {
            ('paper', 'to', 'paper'): GCNConv(-1, 64),
            ('author', 'to', 'paper'): SAGEConv((-1, -1), 64),
            ('paper', 'to', 'author'): GATConv((-1, -1), 64),
        }, aggr=aggr)

    assert len(list(conv.parameters())) > 0
    assert str(conv) == 'HeteroConv(num_relations=3)'

    out = conv(data.x_dict, data.edge_index_dict,
               edge_weight_dict=data.edge_weight_dict)

    assert len(out) == 2
    if aggr is not None:
        assert out['paper'].size() == (50, 64)
        assert out['author'].size() == (30, 64)
    else:
        assert out['paper'].size() == (50, 2, 64)
        assert out['author'].size() == (30, 1, 64)


class CustomConv(MessagePassing):
    def __init__(self, out_channels):
        super().__init__(aggr='add')
        self.lin = Linear(-1, out_channels)

    def forward(self, x, edge_index, pos):
        return self.propagate(edge_index, x=x, pos=pos)

    def message(self, x_j, pos_i, pos_j):
        return self.lin(torch.cat([x_j, pos_i - pos_j], dim=-1))


def test_hetero_conv_with_custom_conv():
    data = HeteroData()
    data['paper'].x = torch.randn(50, 32)
    data['paper'].pos = torch.randn(50, 3)
    data['author'].x = torch.randn(30, 64)
    data['author'].pos = torch.randn(30, 3)
    data['paper', 'paper'].edge_index = get_edge_index(50, 50, 200)
    data['paper', 'author'].edge_index = get_edge_index(50, 30, 100)
    data['author', 'paper'].edge_index = get_edge_index(30, 50, 100)

    conv = HeteroConv({key: CustomConv(64) for key in data.edge_types})
    out = conv(data.x_dict, data.edge_index_dict, data.pos_dict)
    assert len(out) == 2
    assert out['paper'].size() == (50, 64)
    assert out['author'].size() == (30, 64)
