import torch

import torch_geometric.typing
from torch_geometric import EdgeIndex
from torch_geometric.nn import GraphConv
from torch_geometric.testing import is_full_test
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import to_torch_csc_tensor


def test_graph_conv():
    x1 = torch.randn(4, 8)
    x2 = torch.randn(2, 16)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    value = torch.randn(edge_index.size(1))
    adj1 = to_torch_csc_tensor(edge_index, size=(4, 4))
    adj2 = to_torch_csc_tensor(edge_index, value, size=(4, 4))

    conv = GraphConv(8, 32)
    assert str(conv) == 'GraphConv(8, 32)'
    out1 = conv(x1, edge_index)
    assert out1.size() == (4, 32)
    assert torch.allclose(conv(x1, edge_index, size=(4, 4)), out1, atol=1e-6)
    assert torch.allclose(conv(x1, adj1.t()), out1, atol=1e-6)

    assert conv(
        x1,
        EdgeIndex(edge_index, sort_order='col', sparse_size=(4, 4)),
    ).allclose(out1, atol=1e-6)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj3 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 4))
        assert torch.allclose(conv(x1, adj3.t()), out1, atol=1e-6)

    out2 = conv(x1, edge_index, value)
    assert out2.size() == (4, 32)
    assert torch.allclose(conv(x1, edge_index, value, size=(4, 4)), out2,
                          atol=1e-6)
    assert torch.allclose(conv(x1, adj2.t()), out2, atol=1e-6)

    assert conv(
        x1,
        EdgeIndex(edge_index, sort_order='col', sparse_size=(4, 4)),
        value,
    ).allclose(out2, atol=1e-6)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj4 = SparseTensor.from_edge_index(edge_index, value, (4, 4))
        assert torch.allclose(conv(x1, adj4.t()), out2, atol=1e-6)

    if is_full_test():
        jit = torch.jit.script(conv)
        assert torch.allclose(jit(x1, edge_index), out1)
        assert torch.allclose(jit(x1, edge_index, size=(4, 4)), out1)
        assert torch.allclose(jit(x1, edge_index, value), out2)
        assert torch.allclose(jit(x1, edge_index, value, size=(4, 4)), out2)

        if torch_geometric.typing.WITH_TORCH_SPARSE:
            assert torch.allclose(jit(x1, adj3.t()), out1, atol=1e-6)
            assert torch.allclose(jit(x1, adj4.t()), out2, atol=1e-6)

    # Test bipartite message passing:
    adj1 = to_torch_csc_tensor(edge_index, size=(4, 2))
    adj2 = to_torch_csc_tensor(edge_index, value, size=(4, 2))

    conv = GraphConv((8, 16), 32)
    assert str(conv) == 'GraphConv((8, 16), 32)'
    out1 = conv((x1, x2), edge_index)
    assert out1.size() == (2, 32)
    assert torch.allclose(conv((x1, x2), edge_index, size=(4, 2)), out1)
    assert torch.allclose(conv((x1, x2), adj1.t()), out1, atol=1e-6)

    assert conv(
        (x1, x2),
        EdgeIndex(edge_index, sort_order='col', sparse_size=(4, 2)),
    ).allclose(out1, atol=1e-6)

    out2 = conv((x1, None), edge_index, size=(4, 2))
    assert out2.size() == (2, 32)
    assert torch.allclose(conv((x1, None), adj1.t()), out2, atol=1e-6)

    assert conv(
        (x1, None),
        EdgeIndex(edge_index, sort_order='col', sparse_size=(4, 2)),
    ).allclose(out2, atol=1e-6)

    out3 = conv((x1, x2), edge_index, value)
    assert out3.size() == (2, 32)
    assert torch.allclose(conv((x1, x2), edge_index, value, (4, 2)), out3)
    assert torch.allclose(conv((x1, x2), adj2.t()), out3, atol=1e-6)

    assert conv(
        (x1, x2),
        EdgeIndex(edge_index, sort_order='col', sparse_size=(4, 2)),
        value,
    ).allclose(out3, atol=1e-6)

    out4 = conv((x1, None), edge_index, value, size=(4, 2))
    assert out4.size() == (2, 32)
    assert torch.allclose(conv((x1, None), adj2.t()), out4, atol=1e-6)

    assert conv(
        (x1, None),
        EdgeIndex(edge_index, sort_order='col', sparse_size=(4, 2)),
        value,
    ).allclose(out4, atol=1e-6)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj3 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 2))
        adj4 = SparseTensor.from_edge_index(edge_index, value, (4, 2))
        assert torch.allclose(conv((x1, x2), adj3.t()), out1, atol=1e-6)
        assert torch.allclose(conv((x1, None), adj3.t()), out2, atol=1e-6)
        assert torch.allclose(conv((x1, x2), adj3.t()), out1, atol=1e-6)
        assert torch.allclose(conv((x1, None), adj4.t()), out4, atol=1e-6)

    if is_full_test():
        jit = torch.jit.script(conv)
        assert torch.allclose(jit((x1, x2), edge_index), out1)
        assert torch.allclose(jit((x1, x2), edge_index, size=(4, 2)), out1)
        assert torch.allclose(jit((x1, None), edge_index, size=(4, 2)), out2)
        assert torch.allclose(jit((x1, x2), edge_index, value), out3)
        assert torch.allclose(jit((x1, x2), edge_index, value, (4, 2)), out3)
        assert torch.allclose(jit((x1, None), edge_index, value, (4, 2)), out4)

        if torch_geometric.typing.WITH_TORCH_SPARSE:
            assert torch.allclose(jit((x1, x2), adj3.t()), out1, atol=1e-6)
            assert torch.allclose(jit((x1, None), adj3.t()), out2, atol=1e-6)
            assert torch.allclose(jit((x1, x2), adj4.t()), out3, atol=1e-6)
            assert torch.allclose(jit((x1, None), adj4.t()), out4, atol=1e-6)


class EdgeGraphConv(GraphConv):
    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j


def test_inheritance():
    x = torch.randn(4, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    edge_weight = torch.rand(4)

    conv = EdgeGraphConv(8, 16)
    assert conv(x, edge_index, edge_weight).size() == (4, 16)
