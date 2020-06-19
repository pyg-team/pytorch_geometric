import torch
from torch_geometric.nn import GMMConv


def test_gmm_conv():
    in_channels, out_channels = (16, 32)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    num_nodes = edge_index.max().item() + 1
    x = torch.randn((num_nodes, in_channels))
    pseudo = torch.rand((edge_index.size(1), 3))

    conv = GMMConv(in_channels, out_channels, dim=3, kernel_size=25)
    assert conv.__repr__() == 'GMMConv(16, 32)'
    out = conv(x, edge_index, pseudo)
    assert out.size() == (num_nodes, out_channels)

    jit_conv = conv.jittable()
    jit_conv = torch.jit.script(jit_conv)
    assert jit_conv(x, edge_index, pseudo).tolist() == out.tolist()

    conv = GMMConv(in_channels, out_channels, dim=3, kernel_size=25,
                   separate_gaussians=True)
    out = conv(x, edge_index, pseudo)
    assert out.size() == (num_nodes, out_channels)

    jit_conv = conv.jittable()
    jit_conv = torch.jit.script(jit_conv)
    assert jit_conv(x, edge_index, pseudo).tolist() == out.tolist()
