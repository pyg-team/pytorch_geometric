import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor

from torch_geometric.nn import WLConv


def test_wl_conv():
    x1 = torch.tensor([1, 0, 0, 1])
    x2 = F.one_hot(x1).to(torch.float)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    adj_t = SparseTensor.from_edge_index(edge_index).t()

    conv = WLConv()
    assert str(conv) == 'WLConv()'

    out = conv(x1, edge_index)
    assert out.tolist() == [0, 1, 1, 0]
    assert conv(x2, edge_index).tolist() == out.tolist()
    assert conv(x1, adj_t).tolist() == out.tolist()
    assert conv(x2, adj_t).tolist() == out.tolist()

    assert conv.histogram(out).tolist() == [[2, 2]]
    assert torch.allclose(conv.histogram(out, norm=True),
                          torch.tensor([[0.7071, 0.7071]]))
