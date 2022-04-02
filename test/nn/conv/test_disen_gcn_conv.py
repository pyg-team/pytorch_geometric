import torch
from torch_sparse import SparseTensor

from torch_geometric.nn import DisenConv
from torch_geometric.testing import is_full_test


def test_gcn_conv():
    in_channels = 2
    out_channels = 7
    batch_size = 2
    num_factors = 3

    x = torch.randn(6, in_channels)
    edge_index = torch.tensor([[2, 3, 4, 5], [0, 0, 1, 1]])
    row, col = edge_index
    adj1 = SparseTensor(row=row, col=col, sparse_sizes=(6, batch_size))

    conv = DisenConv(in_channels, out_channels, K=num_factors)
    repr_str = f'DisenConv({in_channels}, {out_channels}, [{num_factors}]disentangled_channels=[2, 2, 3])'
    assert conv.__repr__() == repr_str

    out1 = conv(x, edge_index, batch_size)
    assert out1.size() == (batch_size, out_channels)
    assert torch.allclose(conv(x, adj1.t()), out1, atol=1e-6)

    out2 = conv(x, edge_index)
    assert out2.size() == (batch_size, out_channels)
    assert torch.allclose(conv(x, adj1.t()), out2, atol=1e-6)

    if is_full_test():
        t = '(Tensor, Tensor, Optional[int]) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert jit(x, edge_index, batch_size).tolist() == out1.tolist()
        assert jit(x, edge_index).tolist() == out1.tolist()

        t = '(Tensor, SparseTensor,  Optional[int]) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit(x, adj1.t(), batch_size), out1, atol=1e-6)
        assert torch.allclose(jit(x, adj1.t()), out1, atol=1e-6)
