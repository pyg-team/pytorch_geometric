import torch

from torch_geometric.nn import PANConv
from torch_geometric.typing import SparseTensor


def test_pan_conv():
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    row, col = edge_index
    adj1 = SparseTensor(row=row, col=col, sparse_sizes=(4, 4))
    adj2 = adj1.to_torch_sparse_csc_tensor()

    conv = PANConv(16, 32, filter_size=2)
    assert str(conv) == 'PANConv(16, 32, filter_size=2)'
    out1, M1 = conv(x, edge_index)
    assert out1.size() == (4, 32)

    out2, M2 = conv(x, adj1.t())
    assert torch.allclose(out1, out2, atol=1e-6)
    assert torch.allclose(M1.to_dense(), M2.to_dense())

    out3, M3 = conv(x, adj2.t())
    assert torch.allclose(out1, out3, atol=1e-6)
    assert torch.allclose(M1.to_dense(), M3.to_dense())
