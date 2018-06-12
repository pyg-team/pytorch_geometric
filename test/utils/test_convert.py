import torch
import scipy.sparse
import networkx
from torch_geometric.utils import to_scipy_sparse_matrix, to_networkx


def test_to_scipy_sparse_matrix():
    row = torch.tensor([0, 1, 0])
    col = torch.tensor([1, 0, 0])

    adj = to_scipy_sparse_matrix(torch.stack([row, col], dim=0))
    assert isinstance(adj, scipy.sparse.coo_matrix) is True
    assert adj.shape == (2, 2)
    assert adj.row.tolist() == row.tolist()
    assert adj.col.tolist() == col.tolist()
    assert adj.data.tolist() == [1, 1, 1]


def test_to_networkx():
    row = torch.tensor([0, 1, 0])
    col = torch.tensor([1, 0, 0])

    adj = to_networkx(torch.stack([row, col], dim=0))
    assert networkx.to_numpy_matrix(adj).tolist() == [[1, 1], [1, 0]]
