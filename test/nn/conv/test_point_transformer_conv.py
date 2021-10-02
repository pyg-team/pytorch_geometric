
import torch
from torch_sparse import SparseTensor
from torch_geometric.nn import PointTransformerConv
from torch.nn import Sequential, ReLU, Linear


def test_point_transformer_conv():
    x1 = torch.randn(4, 16)
    pos1 = torch.randn(4, 3)

    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    row, col = edge_index
    adj = SparseTensor(row=row, col=col, sparse_sizes=(4, 4))

    conv = PointTransformerConv(in_channels=16, out_channels=32)
    assert conv.__repr__() == (
        'PointTransformerConv(16, 32, local_nn='
        'Linear(in_features=3, out_features=32, bias=True)'
        ', attn_nn=None)')

    out = conv(x1, pos1, edge_index)
    assert out.size() == (4, 32)
    assert torch.allclose(conv(x1, pos1, adj.t()), out, atol=1e-6)

    # test with in_channels as tuple
    x2 = torch.randn(4, 8)
    conv = PointTransformerConv(in_channels=(16, 8), out_channels=32)

    out = conv((x1, x2), pos1, edge_index)
    assert out.size() == (4, 32)
    assert torch.allclose(conv((x1, x2), pos1, adj.t()), out, atol=1e-6)

    # Test with custom nn
    out_channels = 64
    local_nn = Sequential(
        Linear(3, 16),
        ReLU(),
        Linear(16, out_channels)
    )

    attn_nn = Sequential(
        Linear(out_channels, 32),
        ReLU(),
        Linear(32, out_channels)
    )

    conv = PointTransformerConv(
        in_channels=16,
        out_channels=out_channels,
        local_nn=local_nn,
        attn_nn=attn_nn)

    out = conv(x1, pos1, edge_index)
    assert out.size() == (4, 64)
    assert torch.allclose(conv(x1, pos1, adj.t()), out, atol=1e-6)
