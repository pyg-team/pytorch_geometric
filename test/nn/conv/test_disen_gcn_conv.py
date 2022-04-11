import pytest
import torch
from torch_sparse import SparseTensor

from torch_geometric.nn import DisenConv
from torch_geometric.testing import is_full_test


def test_disen_gcn_conv():
    in_channels = 2
    out_channels = 7
    batch_size = 2
    num_factors = 3

    x_src = x = torch.randn(6, in_channels)
    x_dst = x[:batch_size]  # dst nodes are placed first
    edge_index = torch.tensor([[2, 3, 4, 5], [0, 0, 1, 1]])
    row, col = edge_index
    adj_bipartite = SparseTensor(row=row, col=col, sparse_sizes=(6, batch_size))
    adj_full = SparseTensor(row=row, col=col, sparse_sizes=(6, 6))

    conv = DisenConv(in_channels, out_channels, K=num_factors)
    repr_str = (
        f"DisenConv({in_channels}, {out_channels}, "
        f"[{num_factors}]disentangled_channels=[2, 2, 3])"
    )
    assert conv.__repr__() == repr_str

    # All nodes are both src and dst nodes
    out = conv(x, edge_index)
    print(out)
    assert conv(x, edge_index, size=(6, 6)).tolist() == out.tolist()
    assert conv(x, adj_full.t()).tolist() == out.tolist()
    assert conv(x, adj_full.t(), size=(6, 6)).tolist() == out.tolist()

    if is_full_test():
        t = "(Tensor, Tensor, Size) -> Tensor"
        jit = torch.jit.script(conv.jittable(t))
        assert jit(x, edge_index).tolist() == out.tolist()
        assert jit(x, edge_index, size=(6, 6)).tolist() == out.tolist()

        t = "(Tensor, SparseTensor, Size) -> Tensor"
        jit = torch.jit.script(conv.jittable(t))
        assert jit(x, adj_full.t()).tolist() == out.tolist()
        assert jit(x, adj_full.t(), size=(6, 6)).tolist() == out.tolist()

    # src nodes include all dst nodes (in a first-placed manner)
    # and input a tuple of features (x_src, x_dst)
    out1 = conv((x_src, x_dst), edge_index)
    print(out1)
    assert out1.tolist() == out[:batch_size].tolist()
    assert (
        conv((x_src, x_dst), edge_index, size=(6, batch_size)).tolist() == out1.tolist()
    )
    assert (
        conv((x_src, None), edge_index, size=(6, batch_size)).tolist() == out1.tolist()
    )
    with pytest.raises(RuntimeError):  # shape should be (6,2)
        conv((x_src, x_dst), edge_index, size=(6, 6))

    assert conv((x_src, x_dst), adj_bipartite.t()).tolist() == out1.tolist()
    assert conv((x_src, None), adj_bipartite.t()).tolist() == out1.tolist()

    if is_full_test():
        t = "(OptPairTensor, Tensor, Size) -> Tensor"
        jit = torch.jit.script(conv.jittable(t))
        assert jit((x_src, x_dst), edge_index).tolist() == out1.tolist()
        assert jit((x_src, x_dst), edge_index, size=(6, 2)).tolist() == out1.tolist()
        assert jit((x_src, None), edge_index, size=(6, 2)).tolist() == out1.tolist()

        t = "(OptPairTensor, SparseTensor, Size) -> Tensor"
        jit = torch.jit.script(conv.jittable(t))
        assert jit((x_src, x_dst), adj_full.t()).tolist() == out1.tolist()
        assert jit((x_src, x_dst), adj_full.t(), size=(6, 6)).tolist() == out1.tolist()
        assert jit((x_src, None), adj_full.t()).tolist() == out1.tolist()
