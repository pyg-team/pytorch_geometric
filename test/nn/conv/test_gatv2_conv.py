import torch
from torch_sparse import SparseTensor

from torch_geometric.nn import GATv2Conv
from torch_geometric.testing import is_full_test


def test_gatv2_conv():
    x1 = torch.randn(4, 8)
    x2 = torch.randn(2, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    row, col = edge_index
    adj = SparseTensor(row=row, col=col, sparse_sizes=(4, 4))

    conv = GATv2Conv(8, 32, heads=2)
    assert conv.__repr__() == 'GATv2Conv(8, 32, heads=2)'
    out = conv(x1, edge_index)
    assert out.size() == (4, 64)
    assert torch.allclose(conv(x1, edge_index), out)
    assert torch.allclose(
        conv(x1, adj.t()),
        out,
    )

    if is_full_test():
        t = '(Tensor, Tensor, OptTensor, NoneType) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit(x1, edge_index), out)

        t = '(Tensor, SparseTensor, OptTensor, NoneType) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(
            jit(x1, adj.t()),
            out,
        )

    # Test `return_attention_weights`.
    result = conv(x1, edge_index, return_attention_weights=True)
    assert torch.allclose(result[0], out)
    assert result[1][0].size() == (2, 7)
    assert result[1][1].size() == (7, 2)
    assert result[1][1].min() >= 0 and result[1][1].max() <= 1
    assert conv._alpha is None

    result = conv(x1, adj.t(), return_attention_weights=True)
    assert torch.allclose(
        result[0],
        out,
    )
    assert result[1].sizes() == [4, 4, 2] and result[1].nnz() == 7
    assert conv._alpha is None

    if is_full_test():
        t = ('(Tensor, Tensor, OptTensor, bool) -> '
             'Tuple[Tensor, Tuple[Tensor, Tensor]]')
        jit = torch.jit.script(conv.jittable(t))
        result = jit(x1, edge_index, return_attention_weights=True)
        assert torch.allclose(result[0], out)
        assert result[1][0].size() == (2, 7)
        assert result[1][1].size() == (7, 2)
        assert result[1][1].min() >= 0 and result[1][1].max() <= 1
        assert conv._alpha is None

        t = ('(Tensor, SparseTensor, OptTensor, bool) -> '
             'Tuple[Tensor, SparseTensor]')
        jit = torch.jit.script(conv.jittable(t))
        result = jit(x1, adj.t(), return_attention_weights=True)
        assert torch.allclose(
            result[0],
            out,
        )
        assert result[1].sizes() == [4, 4, 2] and result[1].nnz() == 7
        assert conv._alpha is None

    adj = adj.sparse_resize((4, 2))
    out1 = conv((x1, x2), edge_index)
    assert out1.size() == (2, 64)
    assert torch.allclose(conv((x1, x2), edge_index), out1)
    assert torch.allclose(
        conv((x1, x2), adj.t()),
        out1,
    )

    if is_full_test():
        t = '(OptPairTensor, Tensor, OptTensor, NoneType) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit((x1, x2), edge_index), out1)

        t = '(OptPairTensor, SparseTensor, OptTensor, NoneType) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(
            jit((x1, x2), adj.t()),
            out1,
        )


def test_gatv2_conv_with_edge_attr():
    x = torch.randn(4, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 1, 1]])
    edge_weight = torch.randn(edge_index.size(1))
    edge_attr = torch.randn(edge_index.size(1), 4)

    conv = GATv2Conv(8, 32, heads=2, edge_dim=1, fill_value=0.5)
    out = conv(x, edge_index, edge_weight)
    assert out.size() == (4, 64)

    conv = GATv2Conv(8, 32, heads=2, edge_dim=1, fill_value='mean')
    out = conv(x, edge_index, edge_weight)
    assert out.size() == (4, 64)

    conv = GATv2Conv(8, 32, heads=2, edge_dim=4, fill_value=0.5)
    out = conv(x, edge_index, edge_attr)
    assert out.size() == (4, 64)

    conv = GATv2Conv(8, 32, heads=2, edge_dim=4, fill_value='mean')
    out = conv(x, edge_index, edge_attr)
    assert out.size() == (4, 64)
