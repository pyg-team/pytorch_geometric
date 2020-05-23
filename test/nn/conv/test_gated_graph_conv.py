import torch
from torch_geometric.nn import GatedGraphConv


def test_gated_graph_conv():
    in_channels, out_channels = (16, 32)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    num_nodes = edge_index.max().item() + 1
    x = torch.randn((num_nodes, in_channels))

    conv = GatedGraphConv(out_channels, num_layers=3)
    assert conv.__repr__() == 'GatedGraphConv(32, num_layers=3)'
    assert conv(x, edge_index).size() == (num_nodes, out_channels)
    jitcls = conv.jittable(x=x, edge_index=edge_index, full_eval=True)
    jitconv = jitcls(out_channels, num_layers=3)
    jitconv.load_state_dict(conv.state_dict())
    jittedconv = torch.jit.script(jitconv)
    conv.eval()
    jitconv.eval()
    jittedconv.eval()
    assert (torch.abs(conv(x, edge_index) -
            jitconv(x, edge_index)) < 1e-6).all().item()
    assert (torch.abs(conv(x, edge_index) -
            jittedconv(x, edge_index)) < 1e-6).all().item()

    edge_weight = torch.randn(6)
    assert conv(x, edge_index, edge_weight).size() == (num_nodes, out_channels)
    assert (torch.abs(conv(x, edge_index, edge_weight) -
            jitconv(x, edge_index, edge_weight)) < 1e-6).all().item()
    assert (torch.abs(conv(x, edge_index, edge_weight) -
            jittedconv(x, edge_index, edge_weight)) < 1e-6).all().item()
