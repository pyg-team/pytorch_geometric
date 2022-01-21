import torch

from torch_geometric.nn import LEConv


def test_le_conv():
    in_channels, out_channels = (16, 32)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    num_nodes = edge_index.max().item() + 1
    x = torch.randn((num_nodes, in_channels))

    conv = LEConv(in_channels, out_channels)
    assert conv.__repr__() == 'LEConv(16, 32)'
    out = conv(x, edge_index)
    assert out.size() == (num_nodes, out_channels)

    t = '(Tensor, Tensor, OptTensor) -> Tensor'
    jit = torch.jit.script(conv.jittable(t))
    assert jit(x, edge_index).tolist() == out.tolist()
