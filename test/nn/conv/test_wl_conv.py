import torch

from torch_geometric.nn import WLConv
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import one_hot


def test_wl_conv():
    x1 = torch.tensor([1, 0, 0, 1])
    x2 = one_hot(x1)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    adj1 = SparseTensor.from_edge_index(edge_index)
    adj2 = adj1.to_torch_sparse_csc_tensor()

    conv = WLConv()
    assert str(conv) == 'WLConv()'

    out = conv(x1, edge_index)
    assert out.tolist() == [0, 1, 1, 0]
    assert torch.equal(conv(x2, edge_index), out)
    assert torch.equal(conv(x1, adj1.t()), out)
    assert torch.equal(conv(x1, adj2.t()), out)
    assert torch.equal(conv(x2, adj1.t()), out)
    assert torch.equal(conv(x2, adj2.t()), out)

    assert conv.histogram(out).tolist() == [[2, 2]]
    assert torch.allclose(conv.histogram(out, norm=True),
                          torch.tensor([[0.7071, 0.7071]]))
