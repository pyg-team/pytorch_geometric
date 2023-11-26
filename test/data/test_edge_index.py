import torch
from torch import Tensor

from torch_geometric.data.edge_index import EdgeIndex


def test_edge_index():
    adj = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]], sparse_size=(3, 3))
    assert isinstance(adj, EdgeIndex)
    assert isinstance(adj.as_tensor(), Tensor)
    assert adj.sparse_size == (3, 3)
    assert adj.sort_order is None

    out = adj + 1
    assert isinstance(out, Tensor)


def test_edge_index_cat():
    adj1 = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]], sparse_size=(3, 3))
    adj2 = EdgeIndex([[1, 2, 2, 3], [2, 1, 3, 2]], sparse_size=(4, 4))

    out = torch.cat([adj1, adj2], dim=1)
    assert out.size() == (2, 8)
    assert isinstance(out, EdgeIndex)
    assert out.sparse_size == (4, 4)
    assert out.sort_order is None

    out = torch.cat([adj1, adj2], dim=0)
    assert out.size() == (4, 4)
    assert isinstance(out, Tensor)
