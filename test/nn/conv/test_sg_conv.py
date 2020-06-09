import torch
from torch_geometric.nn import SGConv


def test_sg_conv():
    in_channels, out_channels = (16, 32)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    num_nodes = edge_index.max().item() + 1
    x = torch.randn((num_nodes, in_channels))

    conv = SGConv(in_channels, out_channels, K=10, cached=False)
    assert conv.__repr__() == 'SGConv(16, 32, K=10)'
    out = conv(x, edge_index)
    assert out.size() == (num_nodes, out_channels)

    jit_conv = conv.jittable(x=x, edge_index=edge_index)
    jit_conv = torch.jit.script(jit_conv)
    assert jit_conv(x, edge_index).tolist() == out.tolist()

    conv = SGConv(in_channels, out_channels, K=10, cached=True)
    assert conv.__repr__() == 'SGConv(16, 32, K=10)'
    out = conv(x, edge_index)
    out = conv(x, edge_index)
    assert out.size() == (num_nodes, out_channels)

    jit_conv = conv.jittable(x=x, edge_index=edge_index)
    jit_conv = torch.jit.script(jit_conv)
    jit_conv(x, edge_index)
    assert jit_conv(x, edge_index).tolist() == out.tolist()
