
import torch
from torch_sparse import SparseTensor
from torch_geometric.nn import PointTransformerConv


def test_point_transformer_conv():
    x1 = torch.randn(4, 16)
    pos1 = torch.randn(4, 3)

    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    row, col = edge_index
    adj = SparseTensor(row=row, col=col, sparse_sizes=(4, 4))

    conv = PointTransformerConv(in_channels=16, out_channels=32)
    assert conv.__repr__() == (
        'PointTransformerConv(\n'
        '  (delta): Sequential(\n'
        '    (0): Linear(in_features=3, out_features=64, bias=True)\n'
        '    (1): ReLU()\n'
        '    (2): Linear(in_features=64, out_features=32, bias=True)\n'
        '  )\n'
        '  (gamma): Sequential(\n'
        '    (0): Linear(in_features=32, out_features=64, bias=True)\n'
        '    (1): ReLU()\n'
        '    (2): Linear(in_features=64, out_features=32, bias=True)\n'
        '  )\n'
        '  (alpha): Linear(in_features=16, out_features=32, bias=True)\n'
        '  (phi): Linear(in_features=16, out_features=32, bias=True)\n'
        '  (psi): Linear(in_features=16, out_features=32, bias=True)\n'
        ')')

    out = conv(x1, pos1, edge_index)
    assert out.size() == (4, 32)
    assert torch.allclose(conv(x1, pos1, adj.t()), out, atol=1e-6)
