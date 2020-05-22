import torch
from torch_geometric.nn import DNAConv


def test_dna_conv():
    channels = 32
    num_layers = 3
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    num_nodes = edge_index.max().item() + 1
    x = torch.randn((num_nodes, num_layers, channels))

    conv = DNAConv(channels, heads=4, groups=8, dropout=0.5)
    assert conv.__repr__() == 'DNAConv(32, heads=4, groups=8)'
    assert conv(x, edge_index).size() == (num_nodes, channels)
    jitcls = conv.jittable(x=x, edge_index=edge_index)
    jitconv = jitcls(channels, heads=4, groups=8, dropout=0.5)
    jitconv.load_state_dict(conv.state_dict())
    jittedconv = torch.jit.script(jitconv)
    conv.eval()
    jitconv.eval()
    jittedconv.eval()
    assert (torch.abs(conv(x, edge_index) -
            jitconv(x, edge_index)) < 1e-6).all().item()
    assert (torch.abs(conv(x, edge_index) -
            jittedconv(x, edge_index)) < 1e-6).all().item()

    conv = DNAConv(channels, heads=1, groups=1, dropout=0.5)
    assert conv.__repr__() == 'DNAConv(32, heads=1, groups=1)'
    assert conv(x, edge_index).size() == (num_nodes, channels)
    jitcls = conv.jittable(x=x, edge_index=edge_index)
    jitconv = jitcls(channels, heads=1, groups=1, dropout=0.5)
    jitconv.load_state_dict(conv.state_dict())
    jittedconv = torch.jit.script(jitconv)
    conv.eval()
    jitconv.eval()
    jittedconv.eval()
    assert (torch.abs(conv(x, edge_index) -
            jitconv(x, edge_index)) < 1e-6).all().item()
    assert (torch.abs(conv(x, edge_index) -
            jittedconv(x, edge_index)) < 1e-6).all().item()

    conv = DNAConv(channels, heads=1, groups=1, dropout=0.5, cached=True)
    conv(x, edge_index).size() == (num_nodes, channels)
    conv(x, edge_index).size() == (num_nodes, channels)
    jitcls = conv.jittable(x=x, edge_index=edge_index)
    jitconv = jitcls(channels, heads=1, groups=1, dropout=0.5, cached=True)
    jitconv.load_state_dict(conv.state_dict())
    jittedconv = torch.jit.script(jitconv)
    conv.eval()
    jitconv.eval()
    jittedconv.eval()
    assert (torch.abs(conv(x, edge_index) -
            jitconv(x, edge_index)) < 1e-6).all().item()
    assert (torch.abs(conv(x, edge_index) -
            jittedconv(x, edge_index)) < 1e-6).all().item()
