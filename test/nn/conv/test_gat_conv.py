import torch
from torch_geometric.nn import GATConv


def test_gat_conv():
    in_channels, out_channels = (16, 32)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    num_nodes = edge_index.max().item() + 1
    x = torch.randn((num_nodes, in_channels))

    conv = GATConv(in_channels, out_channels, heads=2, dropout=0.5)
    assert conv.__repr__() == 'GATConv(16, 32, heads=2)'
    assert conv(x, edge_index)[0].size() == (num_nodes, 2 * out_channels)
    assert conv((x, x), edge_index)[0].size() == (num_nodes, 2 * out_channels)
    jitcls = conv.jittable(x=x, edge_index=edge_index, full_eval=True)
    jitconv = jitcls(in_channels, out_channels, heads=2, dropout=0.5)
    jitconv.load_state_dict(conv.state_dict())
    jittedconv = torch.jit.script(jitconv)
    conv.eval()
    jitconv.eval()
    jittedconv.eval()
    assert (torch.abs(conv(x, edge_index)[0] -
            jitconv(x, edge_index)[0]) < 1e-6).all().item()
    assert (torch.abs(conv(x, edge_index)[0] -
            jittedconv(x, edge_index)[0]) < 1e-6).all().item()

    out = conv(x, edge_index, return_attention_weights=True)
    assert out[0].size() == (num_nodes, 2 * out_channels)
    assert out[1][1].size() == (edge_index.size(1) + num_nodes, 2)
    assert conv.alpha_save is None

    conv = GATConv(in_channels, out_channels, heads=2, concat=False)
    assert conv(x, edge_index)[0].size() == (num_nodes, out_channels)
    assert conv((x, x), edge_index)[0].size() == (num_nodes, out_channels)
    jitcls = conv.jittable(x=x, edge_index=edge_index, full_eval=True)
    jitconv = jitcls(in_channels, out_channels, heads=2, concat=False)
    jitconv.load_state_dict(conv.state_dict())
    jittedconv = torch.jit.script(jitconv)
    conv.eval()
    jitconv.eval()
    jittedconv.eval()
    assert (torch.abs(conv(x, edge_index)[0] -
            jitconv(x, edge_index)[0]) < 1e-6).all().item()
    assert (torch.abs(conv(x, edge_index)[0] -
            jittedconv(x, edge_index)[0]) < 1e-6).all().item()

    out = conv(x, edge_index, return_attention_weights=True)
    assert out[0].size() == (num_nodes, out_channels)
    assert out[1][1].size() == (edge_index.size(1) + num_nodes, 2)
    assert conv.alpha_save is None
