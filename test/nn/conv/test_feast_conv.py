import torch
from torch_geometric.nn import FeaStConv


def test_feast_conv():
    in_channels, out_channels = (16, 32)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    num_nodes = edge_index.max().item() + 1
    x = torch.randn((num_nodes, in_channels))

    conv = FeaStConv(in_channels, out_channels, heads=2)
    assert conv.__repr__() == 'FeaStConv(16, 32, heads=2)'
    assert conv(x, edge_index).size() == (num_nodes, out_channels)
    jitcls = conv.jittable(x=x, edge_index=edge_index)
    jitconv = jitcls(in_channels, out_channels, heads=2)
    jitconv.load_state_dict(conv.state_dict())
    jittedconv = torch.jit.script(jitconv)
    conv.eval()
    jitconv.eval()
    jittedconv.eval()
    assert (torch.abs(conv(x, edge_index) -
            jitconv(x, edge_index)) < 1e-6).all().item()
    assert (torch.abs(conv(x, edge_index) -
            jittedconv(x, edge_index)) < 1e-6).all().item()
