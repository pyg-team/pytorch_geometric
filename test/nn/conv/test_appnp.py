import torch
from torch_sparse import SparseTensor

from torch_geometric.nn import APPNP
from torch_geometric.testing import is_full_test


def test_appnp():
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    row, col = edge_index
    adj = SparseTensor(row=row, col=col, sparse_sizes=(4, 4))

    conv = APPNP(K=3, alpha=0.1, cached=True)
    assert str(conv) == 'APPNP(K=3, alpha=0.1)'
    out = conv(x, edge_index)
    assert out.size() == (4, 16)
    assert torch.allclose(conv(x, adj.t()), out)

    # Run again to test the cached functionality:
    assert conv._cached_edge_index is not None
    assert conv._cached_adj_t is not None
    assert torch.allclose(conv(x, edge_index), conv(x, adj.t()))

    conv.reset_parameters()
    assert conv._cached_edge_index is None
    assert conv._cached_adj_t is None

    if is_full_test():
        t = '(Tensor, Tensor, OptTensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert jit(x, edge_index).tolist() == out1.tolist()

        t = '(Tensor, SparseTensor, OptTensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit(x, adj.t()), out1, atol=1e-6)


def test_appnp_dropout():
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])

    # With dropout probability of 1.0, the final output equals to alpha * x:
    conv = APPNP(K=2, alpha=0.1, dropout=1.0)
    assert torch.allclose(0.1 * x, conv(x, edge_index))
