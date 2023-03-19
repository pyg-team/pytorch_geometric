import torch
from torch_sparse import SparseTensor

from torch_geometric.nn import MLP, PointGNNConv
from torch_geometric.testing import is_full_test


def test_point_gnn_conv():
    x = torch.randn(6, 8)
    pos = torch.randn(6, 3)
    edge_index = torch.tensor([[0, 1, 1, 1, 2, 5], [1, 2, 3, 4, 3, 4]])
    row, col = edge_index
    adj1 = SparseTensor(row=row, col=col, sparse_sizes=(6, 6))
    adj2 = adj1.to_torch_sparse_csc_tensor()

    conv = PointGNNConv(
        mlp_h=MLP([8, 16, 3]),
        mlp_f=MLP([3 + 8, 16, 8]),
        mlp_g=MLP([8, 16, 8]),
    )
    assert str(conv) == ('PointGNNConv(\n'
                         '  mlp_h=MLP(8, 16, 3),\n'
                         '  mlp_f=MLP(11, 16, 8),\n'
                         '  mlp_g=MLP(8, 16, 8),\n'
                         ')')

    out = conv(x, pos, edge_index)
    assert out.size() == (6, 8)
    assert torch.allclose(conv(x, pos, adj1.t()), out)
    assert torch.allclose(conv(x, pos, adj2.t()), out)

    if is_full_test():
        t = '(Tensor, Tensor, Tensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit(x, pos, edge_index), out)

        t = '(Tensor, Tensor, SparseTensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit(x, pos, adj1.t()), out)
