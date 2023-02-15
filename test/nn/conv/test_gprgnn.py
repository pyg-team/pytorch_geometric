import torch
from torch_sparse import SparseTensor

from torch_geometric.nn import GPRGNN
from torch_geometric.testing import is_full_test


def test_gprgnn():
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    row, col = edge_index
    adj = SparseTensor(row=row, col=col, sparse_sizes=(4, 4))

    conv = GPRGNN(K=2, alpha=0.1, init='PPR', cached=True)
    assert str(conv) == (
        'GPRGNN(K=2, temp=Parameter containing:\n'
        'tensor([0.1000, 0.0900, 0.8100], requires_grad=True))')
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
        assert torch.allclose(jit(x, edge_index), out)

        t = '(Tensor, SparseTensor, OptTensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit(x, adj.t()), out)


def test_gprgnn_dropout():
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    row, col = edge_index
    adj = SparseTensor(row=row, col=col, sparse_sizes=(4, 4))

    # With dropout probability equals to 1.0, alpha equals to 0.0,
    # 'Delta' initialization, the final output equals to x:
    conv = GPRGNN(K=2, alpha=0.0, init='Delta', dropout=1.0)
    assert torch.allclose(x, conv(x, edge_index))
    assert torch.allclose(x, conv(x, adj.t()))

    # With dropout probability equals to 1.0, alpha equals to 1.0,
    # 'Delta' initialization, the final output equals to 0:
    conv = GPRGNN(K=2, alpha=1.0, init='Delta', dropout=1.0)
    assert torch.allclose(torch.zeros_like(x), conv(x, edge_index))
    assert torch.allclose(torch.zeros_like(x), conv(x, adj.t()))

    # With dropout probability equals to 1.0, alpha equals to 2.0,
    # 'Delta' initialization, the final output equals to 0:
    conv = GPRGNN(K=2, alpha=2.0, init='Delta', dropout=1.0)
    assert torch.allclose(torch.zeros_like(x), conv(x, edge_index))
    assert torch.allclose(torch.zeros_like(x), conv(x, adj.t()))
