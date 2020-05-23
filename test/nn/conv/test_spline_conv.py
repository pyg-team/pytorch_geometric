import torch
import torch_geometric
from torch_geometric.nn import SplineConv


def test_spline_conv():
    in_channels, out_channels = (16, 32)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    num_nodes = edge_index.max().item() + 1
    x = torch.randn((num_nodes, in_channels))
    pseudo = torch.rand((edge_index.size(1), 3))

    conv = SplineConv(in_channels, out_channels, dim=3, kernel_size=5)
    assert conv.__repr__() == 'SplineConv(16, 32, dim=3)'
    with torch_geometric.debug():
        assert conv(x, edge_index, pseudo).size() == (num_nodes, out_channels)

    jitcls = conv.jittable(x=x, edge_index=edge_index, pseudo=pseudo)
    jitconv = jitcls(in_channels, out_channels, dim=3, kernel_size=5)
    jitconv.load_state_dict(conv.state_dict())
    jittedconv = torch.jit.script(jitconv)
    conv.eval()
    jitconv.eval()
    jittedconv.eval()
    assert (torch.abs(conv(x, edge_index, pseudo) -
            jitconv(x, edge_index, pseudo)) < 1e-6).all().item()
    assert (torch.abs(conv(x, edge_index, pseudo) -
            jittedconv(x, edge_index, pseudo)) < 1e-6).all().item()
