import torch
from torch_sparse import SparseTensor
from torch_geometric.nn import GATConv


def test_gat_conv():
    x1 = torch.randn(4, 8)
    x2 = torch.randn(2, 16)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    row, col = edge_index
    value = torch.randn(row.size(0))
    adj = SparseTensor(row=row, col=col, value=value, sparse_sizes=(4, 4))

    conv = GATConv(8, 32, heads=2)
    assert conv.__repr__() == 'GATConv(8, 32, heads=2)'
    out = conv(x1, edge_index)
    assert out.size() == (4, 64)
    assert conv(x1, edge_index, size=(4, 4)).tolist() == out.tolist()
    assert torch.allclose(conv(x1, adj.t()), out, atol=1e-6)

    t = '(Tensor, Tensor, Size, NoneType) -> Tensor'
    jit = torch.jit.script(conv.jittable(t))
    assert jit(x1, edge_index).tolist() == out.tolist()
    assert jit(x1, edge_index, size=(4, 4)).tolist() == out.tolist()

    t = '(Tensor, SparseTensor, Size, NoneType) -> Tensor'
    jit = torch.jit.script(conv.jittable(t))
    assert torch.allclose(conv(x1, adj.t()), out, atol=1e-6)

    # Test `return_attention_weights`.
    result = conv(x1, edge_index, return_attention_weights=True)
    assert result[0].tolist() == out.tolist()
    assert result[1][0].size() == (2, 7)
    assert result[1][1].size() == (7, 2)
    assert result[1][1].min() >= 0 and result[1][1].max() <= 1
    assert conv._alpha is None

    result = conv(x1, adj.t(), return_attention_weights=True)
    assert torch.allclose(result[0], out, atol=1e-6)
    assert result[1].sizes() == [4, 4, 2] and result[1].nnz() == 7
    assert conv._alpha is None

    t = '(Tensor, Tensor, Size, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]'
    jit = torch.jit.script(conv.jittable(t))
    result = jit(x1, edge_index, return_attention_weights=True)
    assert result[0].tolist() == out.tolist()
    assert result[1][0].size() == (2, 7)
    assert result[1][1].size() == (7, 2)
    assert result[1][1].min() >= 0 and result[1][1].max() <= 1
    assert conv._alpha is None

    t = '(Tensor, SparseTensor, Size, bool) -> Tuple[Tensor, SparseTensor]'
    jit = torch.jit.script(conv.jittable(t))
    result = jit(x1, adj.t(), return_attention_weights=True)
    assert torch.allclose(result[0], out, atol=1e-6)
    assert result[1].sizes() == [4, 4, 2] and result[1].nnz() == 7
    assert conv._alpha is None

    adj = adj.sparse_resize((4, 2))
    conv = GATConv((8, 16), 32, heads=2)
    assert conv.__repr__() == 'GATConv((8, 16), 32, heads=2)'

    out1 = conv((x1, x2), edge_index)
    out2 = conv((x1, None), edge_index, (4, 2))
    assert out1.size() == (2, 64)
    assert out2.size() == (2, 64)
    assert conv((x1, x2), edge_index, (4, 2)).tolist() == out1.tolist()
    assert torch.allclose(conv((x1, x2), adj.t()), out1, atol=1e-6)
    assert torch.allclose(conv((x1, None), adj.t()), out2, atol=1e-6)

    t = '(OptPairTensor, Tensor, Size, NoneType) -> Tensor'
    jit = torch.jit.script(conv.jittable(t))
    assert jit((x1, x2), edge_index).tolist() == out1.tolist()
    assert jit((x1, x2), edge_index, size=(4, 2)).tolist() == out1.tolist()
    assert jit((x1, None), edge_index, size=(4, 2)).tolist() == out2.tolist()

    t = '(OptPairTensor, SparseTensor, Size, NoneType) -> Tensor'
    jit = torch.jit.script(conv.jittable(t))
    assert torch.allclose(conv((x1, x2), adj.t()), out1, atol=1e-6)
    assert torch.allclose(conv((x1, None), adj.t()), out2, atol=1e-6)
