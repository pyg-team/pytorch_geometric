import torch
from torch_sparse import SparseTensor

from torch_geometric.nn import PointGNNConv, MLP
from torch_geometric.testing import is_full_test

def test_pointgnn_conv():
    x = torch.randn(6, 3)
    edge_index = torch.tensor([[0, 1, 1, 1, 2, 5], [1, 2, 3, 4, 3, 4]])
    row, col = edge_index
    adj = SparseTensor(row=row, col=col, sparse_sizes=(6, 6))
    MLP_h = MLP([3, 64, 3])
    MLP_f = MLP([6, 64, 3])
    MLP_g = MLP([3, 64, 3])

    conv = PointGNNConv(state_channels=3, MLP_h=MLP_h, MLP_f=MLP_f, MLP_g=MLP_g)
    assert conv.__repr__() == 'PointGNNConv(3, lin_h=MLP(3, 64, 3),' \
                        ' lin_f=MLP(6, 64, 3),lin_g=MLP(3, 64, 3))'
    out = conv(x, x, edge_index)

    assert out.size() == (6, 3)
    assert torch.allclose(conv(x, x, adj.t()), out, atol=1e-6)

    if is_full_test:
        t = '(Tensor, Tensor, Tensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert jit(x, x, edge_index).tolist() == out.tolist()

        t = '(Tensor, Tensor, SparseTensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit(x, x, adj.t()), out, atol=1e-5)
