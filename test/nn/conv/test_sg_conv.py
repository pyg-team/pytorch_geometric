import torch
from torch_geometric.nn import SGConv


def test_sg_conv():
    in_channels, out_channels = (16, 32)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    num_nodes = edge_index.max().item() + 1
    x = torch.randn((num_nodes, in_channels))

    conv = SGConv(in_channels, out_channels, K=10, cached=True)
    assert conv.__repr__() == 'SGConv(16, 32, K=10)'
    assert conv(x, edge_index).size() == (num_nodes, out_channels)
    assert conv(x, edge_index).size() == (num_nodes, out_channels)
    conv.reset_parameters()
    jitcls = conv.jittable(x=x, edge_index=edge_index, full_eval=True)
    conv.reset_parameters()
    jitconv = jitcls(in_channels, out_channels, K=10, cached=True)
    jitconv.load_state_dict(conv.state_dict())
    jittedconv = torch.jit.script(jitconv)
    conv.eval()
    jitconv.eval()
    jittedconv.eval()
    assert (torch.abs(conv(x, edge_index) -
            jitconv(x, edge_index)) < 1e-6).all().item()
    assert (torch.abs(conv(x, edge_index) -
            jittedconv(x, edge_index)) < 1e-6).all().item()

    conv = SGConv(in_channels, out_channels, K=10, cached=False)
    assert conv.__repr__() == 'SGConv(16, 32, K=10)'
    assert conv(x, edge_index).size() == (num_nodes, out_channels)
    assert conv(x, edge_index).size() == (num_nodes, out_channels)
    conv.reset_parameters()
    jitcls = conv.jittable(x=x, edge_index=edge_index, full_eval=True)
    conv.reset_parameters()
    jitconv = jitcls(in_channels, out_channels, K=10, cached=False)
    jitconv.load_state_dict(conv.state_dict())
    jittedconv = torch.jit.script(jitconv)
    conv.eval()
    jitconv.eval()
    jittedconv.eval()
    assert (torch.abs(conv(x, edge_index) -
            jitconv(x, edge_index)) < 1e-6).all().item()
    assert (torch.abs(conv(x, edge_index) -
            jittedconv(x, edge_index)) < 1e-6).all().item()
