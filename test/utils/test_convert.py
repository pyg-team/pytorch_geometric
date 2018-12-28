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

    edge_attr = torch.Tensor([1, 2, 3])
    adj = to_scipy_sparse_matrix(torch.stack([row, col], dim=0), edge_attr)
    assert isinstance(adj, scipy.sparse.coo_matrix) is True
    assert adj.shape == (2, 2)
    assert adj.row.tolist() == row.tolist()
    assert adj.col.tolist() == col.tolist()
    assert adj.data.tolist() == edge_attr.tolist()


def test_to_networkx():
    x = torch.Tensor([[1, 2], [3, 4]])
    pos = torch.Tensor([[0, 0], [1, 1]])
    row = torch.tensor([0, 1, 0])
    col = torch.tensor([1, 0, 0])
    edge_attr = torch.Tensor([1, 2, 3])

    G = to_networkx(torch.stack([row, col], dim=0), x, edge_attr, pos)
    assert G.nodes[0]['x'].tolist() == [1, 2]
    assert G.nodes[1]['x'].tolist() == [3, 4]
    assert G.nodes[0]['pos'].tolist() == [0, 0]
    assert G.nodes[1]['pos'].tolist() == [1, 1]
    assert networkx.to_numpy_matrix(G).tolist() == [[3, 1], [2, 0]]

    edge_attr = torch.Tensor([[1, 1], [2, 2], [3, 3]])
    G = to_networkx(torch.stack([row, col], dim=0), edge_attr=edge_attr)
    assert G[0][0]['weight'].tolist() == [3, 3]
    assert G[0][1]['weight'].tolist() == [1, 1]
    assert G[1][0]['weight'].tolist() == [2, 2]
