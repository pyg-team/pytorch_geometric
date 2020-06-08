import torch
from torch_geometric.nn import TAGConv


def test_tag_conv():
    in_channels, out_channels = (16, 32)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    edge_weight = torch.rand(edge_index.size(1))
    num_nodes = edge_index.max().item() + 1
    x = torch.randn((num_nodes, in_channels))

    conv = TAGConv(in_channels, out_channels)
    assert conv.__repr__() == 'TAGConv(16, 32, K=3)'
    out1 = conv(x, edge_index)
    assert out1.size() == (num_nodes, out_channels)
    out2 = conv(x, edge_index, edge_weight)
    assert out2.size() == (num_nodes, out_channels)

    jit_conv = conv.jittable(x=x, edge_index=edge_index)
    jit_conv = torch.jit.script(jit_conv)
    assert jit_conv(x, edge_index).tolist() == out1.tolist()
    assert jit_conv(x, edge_index, edge_weight).tolist() == out2.tolist()

    conv = TAGConv(in_channels, out_channels, normalize=False)
    out = conv(x, edge_index, edge_weight)
    assert out.size() == (num_nodes, out_channels)

    jit_conv = conv.jittable(x=x, edge_index=edge_index,
                             edge_weight=edge_weight)
    jit_conv = torch.jit.script(jit_conv)
    assert jit_conv(x, edge_index, edge_weight).tolist() == out.tolist()
