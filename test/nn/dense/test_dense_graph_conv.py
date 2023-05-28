import pytest
import torch

from torch_geometric.nn import DenseGraphConv, GraphConv
from torch_geometric.testing import is_full_test
from torch_geometric.utils import to_dense_adj


@pytest.mark.parametrize('aggr', ['add', 'mean', 'max'])
def test_dense_graph_conv(aggr):
    channels = 16
    sparse_conv = GraphConv(channels, channels, aggr=aggr)
    dense_conv = DenseGraphConv(channels, channels, aggr=aggr)
    assert str(dense_conv) == 'DenseGraphConv(16, 16)'

    # Ensure same weights and bias.
    dense_conv.lin_rel = sparse_conv.lin_rel
    dense_conv.lin_root = sparse_conv.lin_root

    x = torch.randn((5, channels))
    edge_index = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 4],
                               [1, 2, 0, 2, 0, 1, 4, 3]])

    sparse_out = sparse_conv(x, edge_index)
    assert sparse_out.size() == (5, channels)

    adj = to_dense_adj(edge_index)
    mask = torch.ones(5, dtype=torch.bool)

    dense_out = dense_conv(x, adj, mask)[0]

    assert dense_out.size() == (5, channels)
    assert torch.allclose(sparse_out, dense_out, atol=1e-04)

    if is_full_test():
        jit = torch.jit.script(dense_conv)
        assert torch.allclose(jit(x, adj, mask), dense_out)


@pytest.mark.parametrize('aggr', ['add', 'mean', 'max'])
def test_dense_graph_conv_batch(aggr):
    channels = 16
    sparse_conv = GraphConv(channels, channels, aggr=aggr)
    dense_conv = DenseGraphConv(channels, channels, aggr=aggr)

    # Ensure same weights and bias.
    dense_conv.lin_rel = sparse_conv.lin_rel
    dense_conv.lin_root = sparse_conv.lin_root

    x = torch.randn((5, channels))
    edge_index = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 4],
                               [1, 2, 0, 2, 0, 1, 4, 3]])

    sparse_out = sparse_conv(x, edge_index)
    assert sparse_out.size() == (5, channels)

    x = torch.cat([x, x.new_zeros(1, channels)], dim=0).view(2, 3, channels)
    adj = torch.Tensor([
        [
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ],
        [
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 0],
        ],
    ])
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)

    dense_out = dense_conv(x, adj, mask)
    assert dense_out.size() == (2, 3, channels)
    dense_out = dense_out.view(-1, channels)

    assert torch.allclose(sparse_out, dense_out[:5], atol=1e-04)
    assert dense_out[-1].abs().sum() == 0


@pytest.mark.parametrize('aggr', ['add', 'mean', 'max'])
def test_dense_graph_conv_with_broadcasting(aggr):
    batch_size, num_nodes, channels = 8, 3, 16
    conv = DenseGraphConv(channels, channels, aggr=aggr)

    x = torch.randn(batch_size, num_nodes, channels)
    adj = torch.Tensor([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
    ])

    assert conv(x, adj).size() == (batch_size, num_nodes, channels)
    mask = torch.tensor([1, 1, 1], dtype=torch.bool)
    assert conv(x, adj, mask).size() == (batch_size, num_nodes, channels)
