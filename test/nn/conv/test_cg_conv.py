import torch
from torch_geometric.nn import CGConv


def test_cg_conv():
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    num_nodes = edge_index.max().item() + 1
    x = torch.randn((num_nodes, 16))
    pseudo = torch.rand((edge_index.size(1), 3))

    conv = CGConv(16, 3)
    assert conv.__repr__() == 'CGConv(16, 16, dim=3)'
    out = conv(x, edge_index, pseudo)
    assert out.size() == (num_nodes, 16)

    jit = torch.jit.script(conv.jittable())
    assert jit(x, edge_index, pseudo).tolist() == out.tolist()
