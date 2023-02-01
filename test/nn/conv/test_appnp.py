import torch
from torch_sparse import SparseTensor

from torch_geometric.nn import APPNP
from torch_geometric.testing import is_full_test


def test_appnp():
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    row, col = edge_index
    adj = SparseTensor(row=row, col=col, sparse_sizes=(4, 4))

    conv1 = APPNP(K=3, alpha=0.1, cached=True)
    assert conv1.__repr__() == 'APPNP(K=3, alpha=0.1)'
    out1 = conv1(x, edge_index)
    assert out1.size() == (4, 16)
    out2 = conv1(x, adj.t())
    assert torch.allclose(out1, out2, atol=1e-6)
    assert conv1._cached_edge_index is not None
    assert conv1._cached_adj_t is not None

    # Run again to test the cached functionality
    assert torch.allclose(conv1(x, edge_index), conv1(x, adj.t()), atol=1e-6)

    conv1.reset_parameters()
    assert conv1._cached_edge_index is None
    assert conv1._cached_adj_t is None

    # With dropout probability of 1.0, the final output
    # equals to alpha * x in this APPNP case
    conv2 = APPNP(K=2, alpha=0.1, dropout=1.0)
    assert torch.allclose(0.1 * x, conv2(x, edge_index), atol=1e-6)
    assert torch.allclose(0.1 * x, conv2(x, adj.t()), atol=1e-6)

    if is_full_test():
        t = '(Tensor, Tensor, OptTensor) -> Tensor'
        jit = torch.jit.script(conv1.jittable(t))
        assert jit(x, edge_index).tolist() == out1.tolist()

        t = '(Tensor, SparseTensor, OptTensor) -> Tensor'
        jit = torch.jit.script(conv1.jittable(t))
        assert torch.allclose(jit(x, adj.t()), out1, atol=1e-6)

