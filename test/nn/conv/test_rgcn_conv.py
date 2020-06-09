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
    out = conv(x, edge_index, edge_type)
    assert out.size() == (num_nodes, out_channels)

    jit_conv = conv.jittable(x=x, edge_index=edge_index, edge_type=edge_type)
    jit_conv = torch.jit.script(jit_conv)
    assert jit_conv(x, edge_index, edge_type).tolist() == out.tolist()

    x = None
    conv = RGCNConv(num_nodes, out_channels, num_relations, num_bases=4)
    out = conv(x, edge_index, edge_type)
    assert out.size() == (num_nodes, out_channels)

    jit_conv = conv.jittable(x=x, edge_index=edge_index, edge_type=edge_type)
    jit_conv = torch.jit.script(jit_conv)
    assert jit_conv(x, edge_index, edge_type).tolist() == out.tolist()
