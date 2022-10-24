import torch

from torch_geometric.nn import ChebConv
from torch_geometric.testing import is_full_test


def test_cheb_conv():
    in_channels, out_channels = (16, 32)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    num_nodes = edge_index.max().item() + 1
    edge_weight = torch.rand(edge_index.size(1))
    x = torch.randn((num_nodes, in_channels))

    conv = ChebConv(in_channels, out_channels, K=3)
    assert conv.__repr__() == 'ChebConv(16, 32, K=3, normalization=sym)'
    out1 = conv(x, edge_index)
    assert out1.size() == (num_nodes, out_channels)
    out2 = conv(x, edge_index, edge_weight)
    assert out2.size() == (num_nodes, out_channels)
    out3 = conv(x, edge_index, edge_weight, lambda_max=3.0)
    assert out3.size() == (num_nodes, out_channels)

    if is_full_test():
        jit = torch.jit.script(conv.jittable())
        assert jit(x, edge_index).tolist() == out1.tolist()
        assert jit(x, edge_index, edge_weight).tolist() == out2.tolist()
        assert jit(x, edge_index, edge_weight,
                   lambda_max=torch.tensor(3.0)).tolist() == out3.tolist()

    batch = torch.tensor([0, 0, 1, 1])
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
    num_nodes = edge_index.max().item() + 1
    edge_weight = torch.rand(edge_index.size(1))
    x = torch.randn((num_nodes, in_channels))
    lambda_max = torch.tensor([2.0, 3.0])

    out4 = conv(x, edge_index, edge_weight, batch)
    assert out4.size() == (num_nodes, out_channels)
    out5 = conv(x, edge_index, edge_weight, batch, lambda_max)
    assert out5.size() == (num_nodes, out_channels)

    if is_full_test():
        assert jit(x, edge_index, edge_weight, batch).tolist() == out4.tolist()
        assert jit(x, edge_index, edge_weight, batch,
                   lambda_max).tolist() == out5.tolist()
