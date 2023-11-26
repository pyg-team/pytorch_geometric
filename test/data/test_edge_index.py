import torch

from torch_geometric.data.edge_index import EdgeIndex


def test_edge_index():
    adj = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]], sparse_size=(10, 2))
    assert adj.sparse_size == (10, 2)
    assert adj.sort_order is None

    out = adj + 1
    assert isinstance(out, EdgeIndex)
    assert out.sparse_size == (None, None)
    assert out.sort_order is None

    out = torch.cat([adj, adj], dim=1)
    assert isinstance(out, EdgeIndex)
    assert out.sparse_size == (None, None)
    assert out.sort_order is None
