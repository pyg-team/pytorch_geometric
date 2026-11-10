import torch

from torch_geometric.nn import HypergraphConv


def test_hypergraph_conv_with_more_nodes_than_edges():
    in_channels, out_channels = (16, 32)
    hyperedge_index = torch.tensor([[0, 0, 1, 1, 2, 3], [0, 1, 0, 1, 0, 1]])
    num_nodes = hyperedge_index[0].max().item() + 1
    num_edges = hyperedge_index[1].max().item() + 1
    x = torch.randn((num_nodes, in_channels))
    hyperedge_weight = torch.tensor([1.0, 0.5])
    hyperedge_attr = torch.randn((num_edges, in_channels))

    conv = HypergraphConv(in_channels, out_channels)
    assert str(conv) == 'HypergraphConv(16, 32)'
    out = conv(x, hyperedge_index)
    assert out.size() == (num_nodes, out_channels)
    out = conv(x, hyperedge_index, hyperedge_weight)
    assert out.size() == (num_nodes, out_channels)

    conv = HypergraphConv(in_channels, out_channels, use_attention=True,
                          heads=2)
    out = conv(x, hyperedge_index, hyperedge_attr=hyperedge_attr)
    assert out.size() == (num_nodes, 2 * out_channels)
    out = conv(x, hyperedge_index, hyperedge_weight, hyperedge_attr)
    assert out.size() == (num_nodes, 2 * out_channels)

    conv = HypergraphConv(in_channels, out_channels, use_attention=True,
                          heads=2, concat=False, dropout=0.5)
    out = conv(x, hyperedge_index, hyperedge_weight, hyperedge_attr)
    assert out.size() == (num_nodes, out_channels)


def test_hypergraph_conv_with_more_edges_than_nodes():
    in_channels, out_channels = (16, 32)
    hyperedge_index = torch.tensor([[0, 0, 1, 1, 2, 3, 3, 3, 2, 1, 2],
                                    [0, 1, 2, 1, 2, 1, 0, 3, 3, 4, 4]])
    hyperedge_weight = torch.tensor([1.0, 0.5, 0.8, 0.2, 0.7])
    num_nodes = hyperedge_index[0].max().item() + 1
    x = torch.randn((num_nodes, in_channels))

    conv = HypergraphConv(in_channels, out_channels)
    assert str(conv) == 'HypergraphConv(16, 32)'
    out = conv(x, hyperedge_index)
    assert out.size() == (num_nodes, out_channels)
    out = conv(x, hyperedge_index, hyperedge_weight)
    assert out.size() == (num_nodes, out_channels)


def test_hypergraph_conv_hyperedge_weight():
    in_channels, out_channels = (16, 32)
    hyperedge_index = torch.tensor([[0, 0, 1, 1, 2, 3, 3, 3, 2, 1, 2],
                                    [0, 1, 2, 1, 2, 1, 0, 3, 3, 4, 4]])
    hyperedge_weight = torch.tensor([1.0, 0.5, 0.8, 0.2, 0.7])
    num_nodes = hyperedge_index[0].max().item() + 1
    num_edges = hyperedge_index[1].max().item() + 1
    x = torch.randn((num_nodes, in_channels))
    hyperedge_attr = torch.randn((num_edges, in_channels))

    # X' = D^{-1} H W B^{-1} H^T X Theta:
    conv = HypergraphConv(in_channels, out_channels, bias=False)
    H = torch.zeros(num_nodes, num_edges)
    H[hyperedge_index[0], hyperedge_index[1]] = 1.0
    W = torch.diag(hyperedge_weight)
    D = torch.diag(1.0 / (H @ hyperedge_weight))
    B = torch.diag(1.0 / H.sum(dim=0))
    expected = D @ H @ W @ B @ H.t() @ conv.lin(x)
    out = conv(x, hyperedge_index, hyperedge_weight)
    assert torch.allclose(out, expected, atol=1e-6)

    # Weights of shape `[num_edges, 1]` are supported as well:
    out = conv(x, hyperedge_index, hyperedge_weight.view(-1, 1))
    assert torch.allclose(out, expected, atol=1e-6)

    # Rescaling all hyperedge weights does not change the output:
    out = conv(x, hyperedge_index, 3.0 * hyperedge_weight)
    assert torch.allclose(out, expected, atol=1e-6)

    conv = HypergraphConv(in_channels, out_channels, use_attention=True,
                          heads=2).eval()
    out1 = conv(x, hyperedge_index, hyperedge_weight, hyperedge_attr)
    out2 = conv(x, hyperedge_index, 3.0 * hyperedge_weight, hyperedge_attr)
    assert torch.allclose(out1, out2, atol=1e-6)
    out3 = conv(x, hyperedge_index, torch.ones(num_edges), hyperedge_attr)
    assert not torch.allclose(out1, out3, atol=1e-6)
