import torch

from torch_geometric import EdgeIndex
from torch_geometric.contrib.nn.conv.res_graph_conv import res_gconv_norm
from torch_geometric.typing import SparseTensor


def test_res_gconv_norm():
    # we first make a minimal example for all cases of the
    # sparse tensors

    # this is fo
    edge_index = torch.tensor([[0, 1, 0, 1], [0, 0, 1, 1]])
    edge_weight = torch.tensor([1.0, 1.0, 1.0, 1.0])
    expected_edge_index = torch.tensor([[0, 1, 0, 1], [0, 0, 1, 1]])
    expected_edge_weight = 0.5 * torch.tensor([1.0, 1.0, 1.0, 1.0])
    edge_index_out, edge_weight_out = res_gconv_norm(edge_index, edge_weight,
                                                     add_self_loops=False)
    assert torch.equal(edge_index_out, expected_edge_index)
    assert torch.allclose(expected_edge_weight, edge_weight_out)
    #
    edge_index = EdgeIndex(
        [[0, 1, 1, 2], [1, 0, 2, 1]],
        sparse_size=(3, 3),
        sort_order='row',
        is_undirected=True,
        device='cpu',
    )
    edge_index.to_sparse_tensor()
    print(isinstance(edge_index, SparseTensor))


"""
def test_gcn2_conv():
    x = torch.randn(4, 16)
    x_0 = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    value = torch.rand(edge_index.size(1))
    adj1 = to_torch_csc_tensor(edge_index, size=(4, 4))
    adj2 = to_torch_csc_tensor(edge_index, value, size=(4, 4))

    conv = GCN2Conv(16, alpha=0.2)
    assert str(conv) == 'GCN2Conv(16, alpha=0.2, beta=1.0)'
    out1 = conv(x, x_0, edge_index)
    assert out1.size() == (4, 16)
    assert torch.allclose(conv(x, x_0, adj1.t()), out1, atol=1e-6)
    out2 = conv(x, x_0, edge_index, value)
    assert out2.size() == (4, 16)
    assert torch.allclose(conv(x, x_0, adj2.t()), out2, atol=1e-6)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj3 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 4))
        adj4 = SparseTensor.from_edge_index(edge_index, value, (4, 4))
        assert torch.allclose(conv(x, x_0, adj3.t()), out1, atol=1e-6)
        assert torch.allclose(conv(x, x_0, adj4.t()), out2, atol=1e-6)

    if is_full_test():
        jit = torch.jit.script(conv)
        assert torch.allclose(jit(x, x_0, edge_index), out1, atol=1e-6)
        assert torch.allclose(jit(x, x_0, edge_index, value), out2, atol=1e-6)

        if torch_geometric.typing.WITH_TORCH_SPARSE:
            assert torch.allclose(jit(x, x_0, adj3.t()), out1, atol=1e-6)
            assert torch.allclose(jit(x, x_0, adj4.t()), out2, atol=1e-6)

    conv.cached = True
    conv(x, x_0, edge_index)
    assert conv._cached_edge_index is not None
    assert torch.allclose(conv(x, x_0, edge_index), out1, atol=1e-6)
    assert torch.allclose(conv(x, x_0, adj1.t()), out1, atol=1e-6)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        conv(x, x_0, adj3.t())
        assert conv._cached_adj_t is not None
        assert torch.allclose(conv(x, x_0, adj3.t()), out1, atol=1e-6)
"""
