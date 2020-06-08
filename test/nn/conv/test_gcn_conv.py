import torch
from torch_geometric.nn import GCNConv


def test_gcn_conv():
    in_channels, out_channels = (16, 32)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    num_nodes = edge_index.max().item() + 1
    edge_weight = torch.rand(edge_index.size(1))
    x = torch.randn((num_nodes, in_channels))

    conv = GCNConv(in_channels, out_channels)
    assert conv.__repr__() == 'GCNConv(16, 32)'
    out = conv(x, edge_index)
    assert out.size() == (num_nodes, out_channels)
    assert conv(x, edge_index, edge_weight).size() == (num_nodes, out_channels)

    jit_conv = conv.jittable(x=x, edge_index=edge_index)
    jit_conv = torch.jit.script(jit_conv)
    jit_out = jit_conv(x, edge_index)
    assert jit_out.tolist() == out.tolist()

    conv = GCNConv(in_channels, out_channels, cached=True)
    out = conv(x, edge_index)
    assert out.size() == (num_nodes, out_channels)
    assert conv(x, edge_index, edge_weight).size() == (num_nodes, out_channels)

    jit_conv = conv.jittable(x=x, edge_index=edge_index)
    jit_conv = torch.jit.script(jit_conv)
    jit_out = jit_conv(x, edge_index)
    assert jit_out.tolist() == out.tolist()


def test_sparse_gcn_conv():
    in_channels, out_channels = (16, 32)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    num_nodes = edge_index.max().item() + 1
    x = torch.sparse_coo_tensor(torch.tensor([[0, 0], [0, 1]]),
                                torch.Tensor([1, 1]),
                                torch.Size([num_nodes, in_channels]))

    conv = GCNConv(in_channels, out_channels)
    assert conv(x, edge_index).size() == (num_nodes, out_channels)


def test_static_gcn_conv():
    in_channels, out_channels = (16, 32)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    num_nodes = edge_index.max().item() + 1
    x = torch.randn((4, num_nodes, in_channels))

    conv = GCNConv(in_channels, out_channels, node_dim=1)
    out = conv(x, edge_index)
    assert out.size() == (4, num_nodes, out_channels)

    jit_conv = conv.jittable(x=x, edge_index=edge_index)
    jit_conv = torch.jit.script(jit_conv)
    jit_out = jit_conv(x, edge_index)
    assert jit_out.tolist() == out.tolist()
