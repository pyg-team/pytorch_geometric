import torch

from torch_geometric.data import Batch, Data
from torch_geometric.nn import ChebConv
from torch_geometric.testing import is_full_test


def test_cheb_conv():
    in_channels, out_channels = (16, 32)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    num_nodes = edge_index.max().item() + 1
    edge_weight = torch.rand(edge_index.size(1))
    x = torch.randn((num_nodes, in_channels))

    conv = ChebConv(in_channels, out_channels, K=3)
    assert str(conv) == 'ChebConv(16, 32, K=3, normalization=sym)'
    out1 = conv(x, edge_index)
    assert out1.size() == (num_nodes, out_channels)
    out2 = conv(x, edge_index, edge_weight)
    assert out2.size() == (num_nodes, out_channels)
    out3 = conv(x, edge_index, edge_weight, lambda_max=3.0)
    assert out3.size() == (num_nodes, out_channels)

    if is_full_test():
        jit = torch.jit.script(conv.jittable())
        assert torch.allclose(jit(x, edge_index), out1)
        assert torch.allclose(jit(x, edge_index, edge_weight), out2)
        assert torch.allclose(
            jit(x, edge_index, edge_weight, lambda_max=torch.tensor(3.0)),
            out3)

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
        assert torch.allclose(jit(x, edge_index, edge_weight, batch), out4)
        assert torch.allclose(
            jit(x, edge_index, edge_weight, batch, lambda_max), out5)


def test_cheb_conv_batch():
    x1 = torch.randn(4, 8)
    edge_index1 = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    edge_weight1 = torch.rand(edge_index1.size(1))
    data1 = Data(x=x1, edge_index=edge_index1, edge_weight=edge_weight1)

    x2 = torch.randn(3, 8)
    edge_index2 = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    edge_weight2 = torch.rand(edge_index2.size(1))
    data2 = Data(x=x2, edge_index=edge_index2, edge_weight=edge_weight2)

    conv = ChebConv(8, 16, K=2)

    out1 = conv(x1, edge_index1, edge_weight1)
    out2 = conv(x2, edge_index2, edge_weight2)

    batch = Batch.from_data_list([data1, data2])
    out = conv(batch.x, batch.edge_index, batch.edge_weight, batch.batch)

    assert out.size() == (7, 16)
    assert torch.allclose(out1, out[:4])
    assert torch.allclose(out2, out[4:])
