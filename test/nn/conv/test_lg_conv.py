import torch
from torch_sparse import SparseTensor
from torch_geometric.nn import LGConv


def test_light_conv():
    x1 = torch.randn(4, 8)
    x2 = torch.randn(2, 16)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    row, col = edge_index
    value = torch.randn(edge_index.size(1))
    adj1 = SparseTensor(row=row, col=col, sparse_sizes=(4, 4))
    adj2 = SparseTensor(row=row, col=col, value=value, sparse_sizes=(4, 4))

    conv = LGConv()
    assert conv.__repr__() == 'LGConv()'
    out11 = conv(x1, edge_index)
    assert out11.size() == (4, 32)
    assert conv(x1, edge_index, size=(4, 4)).tolist() == out11.tolist()
    assert conv(x1, adj1.t()).tolist() == out11.tolist()

    t = '(Tensor, Tensor, Size) -> Tensor'
    jit = torch.jit.script(conv.jittable(t))
    assert jit(x1, edge_index).tolist() == out11.tolist()
    assert jit(x1, edge_index, size=(4, 4)).tolist() == out11.tolist()