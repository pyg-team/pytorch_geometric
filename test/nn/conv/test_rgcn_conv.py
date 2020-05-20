import torch
from torch_geometric.nn import RGCNConv


def test_rgcn_conv():
    in_channels, out_channels = (16, 32)
    num_relations = 8
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    num_nodes = edge_index.max().item() + 1
    x = torch.randn((num_nodes, in_channels))
    edge_type = torch.randint(0, num_relations, (edge_index.size(1), ))

    conv = RGCNConv(in_channels, out_channels, num_relations, num_bases=4)
    assert conv.__repr__() == 'RGCNConv(16, 32, num_relations=8)'
    assert conv(x, edge_index, edge_type).size() == (num_nodes, out_channels)
    jitcls = conv.jittable(x=x, edge_index=edge_index, edge_type=edge_type)
    jitconv = jitcls(in_channels, out_channels, num_relations, num_bases=4)
    jitconv.load_state_dict(conv.state_dict())
    jittedconv = torch.jit.script(jitconv)
    conv.eval()
    jitconv.eval()
    jittedconv.eval()
    assert (torch.abs(conv(x, edge_index, edge_type) -
            jitconv(x, edge_index, edge_type)) < 1e-6).all().item()
    assert (torch.abs(conv(x, edge_index, edge_type) -
            jittedconv(x, edge_index, edge_type)) < 1e-6).all().item()

    x = None
    conv = RGCNConv(num_nodes, out_channels, num_relations, num_bases=4)
    assert conv(x, edge_index, edge_type).size() == (num_nodes, out_channels)
    jitcls = conv.jittable(x=x, edge_index=edge_index, edge_type=edge_type)
    jitconv = jitcls(num_nodes, out_channels, num_relations, num_bases=4)
    jitconv.load_state_dict(conv.state_dict())
    jittedconv = torch.jit.script(jitconv)
    conv.eval()
    jitconv.eval()
    jittedconv.eval()
    assert (torch.abs(conv(x, edge_index, edge_type) -
            jitconv(x, edge_index, edge_type)) < 1e-6).all().item()
    assert (torch.abs(conv(x, edge_index, edge_type) -
            jittedconv(x, edge_index, edge_type)) < 1e-6).all().item()
