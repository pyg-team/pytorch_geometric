import torch
from torch_geometric.nn import CGConv


def test_cg_conv():
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    num_nodes = edge_index.max().item() + 1
    x = torch.randn((num_nodes, 16))
    pseudo = torch.rand((edge_index.size(1), 3))

    conv = CGConv(16, 3)
    assert conv.__repr__() == 'CGConv(16, 16, dim=3)'
    assert conv(x, edge_index, pseudo).size() == (num_nodes, 16)
    jitcls = conv.jittable(x=x, edge_index=edge_index, edge_attr=pseudo)
    jitconv = jitcls(16, 3)
    jitconv.load_state_dict(conv.state_dict())
    jittedconv = torch.jit.script(jitconv)
    conv.eval()
    jitconv.eval()
    jittedconv.eval()
    assert (torch.abs(conv(x, edge_index, pseudo) -
            jitconv(x, edge_index, pseudo)) < 1e-6).all().item()
    assert (torch.abs(conv(x, edge_index, pseudo) -
            jittedconv(x, edge_index, pseudo)) < 1e-6).all().item()
