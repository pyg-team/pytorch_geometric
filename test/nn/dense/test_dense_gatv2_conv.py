import pytest
import torch

from torch_geometric.nn import DenseGATv2Conv, GATv2Conv
from torch_geometric.testing import is_full_test


@pytest.mark.parametrize('heads', [1, 4])
@pytest.mark.parametrize('concat', [True, False])
@pytest.mark.parametrize('share_weights', [False, True])
def test_dense_gatv2_conv(heads, concat, share_weights):
    channels = 16
    sparse_conv = GATv2Conv(channels, channels, heads=heads, concat=concat,
                            share_weights=share_weights)
    dense_conv = DenseGATv2Conv(channels, channels, heads=heads, concat=concat,
                                share_weights=share_weights)
    assert str(dense_conv) == f'DenseGATv2Conv(16, 16, heads={heads})'

    # Ensure same weights and bias:
    dense_conv.lin_l = sparse_conv.lin_l
    dense_conv.lin_r = sparse_conv.lin_r
    dense_conv.att = sparse_conv.att
    dense_conv.bias = sparse_conv.bias

    x = torch.randn((5, channels))
    edge_index = torch.tensor([[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]])

    sparse_out = sparse_conv(x, edge_index)

    x = torch.cat([x, x.new_zeros(1, channels)], dim=0).view(2, 3, channels)
    adj = torch.tensor([
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
    ])
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)

    dense_out = dense_conv(x, adj, mask)

    if is_full_test():
        jit = torch.jit.script(dense_conv)
        assert torch.allclose(jit(x, adj, mask), dense_out)

    assert dense_out[1, 2].abs().sum() == 0
    dense_out = dense_out.view(6, dense_out.size(-1))[:-1]
    assert torch.allclose(sparse_out, dense_out, atol=1e-4)


def test_dense_gatv2_conv_with_broadcasting():
    batch_size, num_nodes, channels = 8, 3, 16
    conv = DenseGATv2Conv(channels, channels, heads=4)

    x = torch.randn(batch_size, num_nodes, channels)
    adj = torch.tensor([
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ])

    assert conv(x, adj).size() == (batch_size, num_nodes, 64)
    mask = torch.tensor([1, 1, 1], dtype=torch.bool)
    assert conv(x, adj, mask).size() == (batch_size, num_nodes, 64)


def test_dense_gatv2_conv_share_weights():
    channels = 16
    conv = DenseGATv2Conv(channels, channels, heads=2, share_weights=True)
    assert conv.lin_l is conv.lin_r
