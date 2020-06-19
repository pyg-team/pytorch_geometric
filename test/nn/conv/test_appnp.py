import torch
from torch_geometric.nn import APPNP


def test_appnp():
    in_channels, out_channels = (16, 32)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    edge_weight = torch.rand(edge_index.size(1))
    num_nodes = edge_index.max().item() + 1
    x = torch.randn((num_nodes, in_channels))

    lin = torch.nn.Linear(in_channels, out_channels)
    conv = APPNP(K=10, alpha=0.1)
    assert conv.__repr__() == 'APPNP(K=10, alpha=0.1)'
    out1 = conv(lin(x), edge_index)
    assert out1.size() == (num_nodes, out_channels)
    out2 = conv(lin(x), edge_index, edge_weight)
    assert out2.size() == (num_nodes, out_channels)

    jit = torch.jit.script(conv.jittable())
    assert jit(lin(x), edge_index).tolist() == out1.tolist()
    assert jit(lin(x), edge_index, edge_weight).tolist() == out2.tolist()
