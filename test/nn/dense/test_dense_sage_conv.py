import torch
from torch_geometric.nn import SAGEConv, DenseSAGEConv


def test_dense_sage_conv():
    channels = 16
    sparse_conv = SAGEConv(channels, channels)
    dense_conv = DenseSAGEConv(channels, channels)
    assert dense_conv.__repr__() == 'DenseSAGEConv(16, 16)'

    # Ensure same weights (identity) and bias (ones).
    index = torch.arange(0, channels, dtype=torch.long)
    sparse_conv.weight.data.fill_(0)
    sparse_conv.weight.data[index, index] = 1
    sparse_conv.bias.data.fill_(1)
    dense_conv.weight.data.fill_(0)
    dense_conv.weight.data[index, index] = 1
    dense_conv.bias.data.fill_(1)

    x = torch.randn((5, channels))
    edge_index = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 4],
                               [1, 2, 0, 2, 0, 1, 4, 3]])

    sparse_out = sparse_conv(x, edge_index)

    x = torch.cat([x, x.new_zeros(1, channels)], dim=0).view(2, 3, channels)
    adj = torch.Tensor([
        [[0, 1, 1], [1, 0, 1], [1, 1, 0]],
        [[0, 1, 0], [1, 0, 0], [0, 0, 0]],
    ])
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.uint8)

    dense_out = dense_conv(x, adj, mask)
    assert dense_out.size() == (2, 3, channels)
    dense_out = dense_out.view(6, channels)[:-1]
    assert torch.allclose(sparse_out, dense_out, atol=1e-04)
