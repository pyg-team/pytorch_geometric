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
from torch_geometric.testing import get_random_edge_index


@pytest.mark.parametrize('aggr', ['sum', 'mean', 'min', 'max', 'cat', None])
def test_hetero_conv(aggr):
    data = HeteroData()
    data['paper'].x = torch.randn(50, 32)
    data['author'].x = torch.randn(30, 64)
    data['paper', 'paper'].edge_index = get_random_edge_index(50, 50, 200)
    data['paper', 'author'].edge_index = get_random_edge_index(50, 30, 100)
    data['paper', 'author'].edge_attr = torch.randn(100, 3)
    data['author', 'paper'].edge_index = get_random_edge_index(30, 50, 100)
    data['paper', 'paper'].edge_weight = torch.rand(200)

    # Unspecified edge types should be ignored:
    data['author', 'author'].edge_index = get_random_edge_index(30, 30, 100)

    conv = HeteroConv(
        {
            ('paper', 'to', 'paper'):
            GCNConv(-1, 64),
            ('author', 'to', 'paper'):
            SAGEConv((-1, -1), 64),
            ('paper', 'to', 'author'):
            GATConv((-1, -1), 64, edge_dim=3, add_self_loops=False),
        }, aggr=aggr)

    assert len(list(conv.parameters())) > 0
    assert str(conv) == 'HeteroConv(num_relations=3)'

    out = conv(data.x_dict, data.edge_index_dict, data.edge_attr_dict,
               edge_weight_dict=data.edge_weight_dict)

    assert len(out) == 2
    if aggr == 'cat':
        assert out['paper'].size() == (50, 128)
        assert out['author'].size() == (30, 64)
    elif aggr is not None:
        assert out['paper'].size() == (50, 64)
        assert out['author'].size() == (30, 64)
    else:
        assert out['paper'].size() == (50, 2, 64)
        assert out['author'].size() == (30, 1, 64)


class CustomConv(MessagePassing):
    def __init__(self, out_channels):
        super().__init__(aggr='add')
        self.lin = Linear(-1, out_channels)

    def forward(self, x, edge_index, y, z):
        return self.propagate(edge_index, x=x, y=y, z=z)

    def message(self, x_j, y_j, z_j):
        return self.lin(torch.cat([x_j, y_j, z_j], dim=-1))


def test_hetero_conv_with_custom_conv():
    data = HeteroData()
    data['paper'].x = torch.randn(50, 32)
    data['paper'].y = torch.randn(50, 3)
    data['paper'].z = torch.randn(50, 3)
    data['author'].x = torch.randn(30, 64)
    data['author'].y = torch.randn(30, 3)
    data['author'].z = torch.randn(30, 3)
    data['paper', 'paper'].edge_index = get_random_edge_index(50, 50, 200)
    data['paper', 'author'].edge_index = get_random_edge_index(50, 30, 100)
    data['author', 'paper'].edge_index = get_random_edge_index(30, 50, 100)

    conv = HeteroConv({key: CustomConv(64) for key in data.edge_types})
    # Test node `args_dict` and `kwargs_dict` with `y_dict` and `z_dict`:
    out = conv(data.x_dict, data.edge_index_dict, data.y_dict,
               z_dict=data.z_dict)
    assert len(out) == 2
    assert out['paper'].size() == (50, 64)
    assert out['author'].size() == (30, 64)


class MessagePassingLoops(MessagePassing):
    def __init__(self):
        super().__init__()
        self.add_self_loops = True


def test_hetero_conv_self_loop_error():
    HeteroConv({('a', 'to', 'a'): MessagePassingLoops()})
    with pytest.raises(ValueError, match="incorrect message passing"):
        HeteroConv({('a', 'to', 'b'): MessagePassingLoops()})


def test_hetero_conv_with_dot_syntax_node_types():
    data = HeteroData()
    data['src.paper'].x = torch.randn(50, 32)
    data['author'].x = torch.randn(30, 64)
    edge_index = get_random_edge_index(50, 50, 200)
    data['src.paper', 'src.paper'].edge_index = edge_index
    data['src.paper', 'author'].edge_index = get_random_edge_index(50, 30, 100)
    data['author', 'src.paper'].edge_index = get_random_edge_index(30, 50, 100)
    data['src.paper', 'src.paper'].edge_weight = torch.rand(200)

    conv = HeteroConv({
        ('src.paper', 'to', 'src.paper'):
        GCNConv(-1, 64),
        ('author', 'to', 'src.paper'):
        SAGEConv((-1, -1), 64),
        ('src.paper', 'to', 'author'):
        GATConv((-1, -1), 64, add_self_loops=False),
    })

    assert len(list(conv.parameters())) > 0
    assert str(conv) == 'HeteroConv(num_relations=3)'

    out = conv(data.x_dict, data.edge_index_dict,
               edge_weight_dict=data.edge_weight_dict)

    assert len(out) == 2
    assert out['src.paper'].size() == (50, 64)
    assert out['author'].size() == (30, 64)
