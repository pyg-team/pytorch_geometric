import itertools

import pytest
import torch

from torch_geometric.nn import DenseGATConv, GATConv
from torch_geometric.testing import is_full_test
from torch_geometric.utils import to_dense_adj


@pytest.mark.parametrize('heads, concat',
                         list(itertools.product([1, 4], [True, False])))
def test_dense_gat_conv(heads, concat):
    channels = 16
    sparse_conv = GATConv(channels, channels, heads=heads, concat=concat)
    dense_conv = DenseGATConv(channels, channels, heads=heads, concat=concat)
    assert str(dense_conv) == 'DenseGATConv(16, 16)'

    # Ensure same weights and bias.
    dense_conv.lin = sparse_conv.lin_src
    dense_conv.att_src = sparse_conv.att_src
    dense_conv.att_dst = sparse_conv.att_dst
    dense_conv.bias = sparse_conv.bias

    x = torch.ones((5, channels))

    edge_index = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 4],
                               [1, 2, 0, 2, 0, 1, 4, 3]])

    sparse_out = sparse_conv(x, edge_index)

    adj = to_dense_adj(edge_index)
    mask = torch.ones(5, dtype=torch.bool)

    dense_out = dense_conv(x, adj, mask)[0]

    assert dense_out.size() == sparse_out.size()
    assert torch.allclose(sparse_out, dense_out, atol=1e-04)

    if is_full_test():
        jit = torch.jit.script(dense_conv)
        assert torch.allclose(jit(x, adj, mask), dense_out)


@pytest.mark.parametrize('heads, concat',
                         list(itertools.product([1, 4], [True, False])))
def test_dense_gat_conv_with_broadcasting(heads, concat):
    batch_size, num_nodes, channels = 8, 3, 16
    conv = DenseGATConv(channels, channels, heads=heads, concat=concat)

    x = torch.randn(batch_size, num_nodes, channels)
    adj = torch.Tensor([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
    ])

    if concat:
        out_channels = channels * heads
    else:
        out_channels = channels

    assert conv(x, adj).size() == (batch_size, num_nodes, out_channels)
    mask = torch.tensor([1, 1, 1], dtype=torch.bool)
    assert conv(x, adj, mask).size() == (batch_size, num_nodes, out_channels)
