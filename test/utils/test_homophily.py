import pytest
import torch
from torch_sparse import SparseTensor

from torch_geometric.utils import homophily


def test_homophily():
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 0, 4]])
    y = torch.tensor([0, 0, 0, 0, 1])
    batch = torch.tensor([0, 0, 0, 1, 1])
    row, col = edge_index
    adj = SparseTensor(row=row, col=col, sparse_sizes=(5, 5))

    assert pytest.approx(homophily(edge_index, y, method='edge')) == 0.75
    assert pytest.approx(homophily(adj, y, method='edge')) == 0.75
    assert homophily(edge_index, y, batch, method='edge').tolist() == [1., 0.]

    assert pytest.approx(homophily(edge_index, y, method='node')) == 0.6
    assert pytest.approx(homophily(adj, y, method='node')) == 0.6
    assert homophily(edge_index, y, batch, method='node').tolist() == [1., 0.]
