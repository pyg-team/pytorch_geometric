import torch
import scipy.sparse
import networkx as nx
from torch_sparse import coalesce
from torch_geometric.data import Data
from torch_geometric.utils import to_scipy_sparse_matrix, to_networkx
from torch_geometric.utils import from_scipy_sparse_matrix, from_networkx
from torch_geometric.utils import subgraph


def test_to_scipy_sparse_matrix():
    edge_index = torch.tensor([[0, 1, 0], [1, 0, 0]])

    adj = to_scipy_sparse_matrix(edge_index)
    assert isinstance(adj, scipy.sparse.coo_matrix) is True
    assert adj.shape == (2, 2)
    assert adj.row.tolist() == edge_index[0].tolist()
    assert adj.col.tolist() == edge_index[1].tolist()
    assert adj.data.tolist() == [1, 1, 1]

    edge_attr = torch.Tensor([1, 2, 3])
    adj = to_scipy_sparse_matrix(edge_index, edge_attr)
    assert isinstance(adj, scipy.sparse.coo_matrix) is True
    assert adj.shape == (2, 2)
    assert adj.row.tolist() == edge_index[0].tolist()
    assert adj.col.tolist() == edge_index[1].tolist()
    assert adj.data.tolist() == edge_attr.tolist()


def test_from_scipy_sparse_matrix():
    edge_index = torch.tensor([[0, 1, 0], [1, 0, 0]])
    adj = to_scipy_sparse_matrix(edge_index)

    out = from_scipy_sparse_matrix(adj)
    assert out[0].tolist() == edge_index.tolist()
    assert out[1].tolist() == [1, 1, 1]


def test_to_networkx():
    x = torch.Tensor([[1, 2], [3, 4]])
    pos = torch.Tensor([[0, 0], [1, 1]])
    edge_index = torch.tensor([[0, 1, 0], [1, 0, 0]])
    edge_attr = torch.Tensor([1, 2, 3])
    data = Data(x=x, pos=pos, edge_index=edge_index, weight=edge_attr)

    G = to_networkx(data, node_attrs=['x', 'pos'], edge_attrs=['weight'])
    assert G.nodes[0]['x'] == [1, 2]
    assert G.nodes[1]['x'] == [3, 4]
    assert G.nodes[0]['pos'] == [0, 0]
    assert G.nodes[1]['pos'] == [1, 1]
    assert nx.to_numpy_matrix(G).tolist() == [[3, 1], [2, 0]]


def test_from_networkx():
    x = torch.Tensor([[1, 2], [3, 4]])
    pos = torch.Tensor([[0, 0], [1, 1]])
    edge_index = torch.tensor([[0, 1, 0], [1, 0, 0]])
    edge_attr = torch.Tensor([1, 2, 3])
    data = Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr)
    G = to_networkx(data, node_attrs=['x', 'pos'], edge_attrs=['edge_attr'])

    data = from_networkx(G)
    assert len(data) == 4
    assert data.x.tolist() == x.tolist()
    assert data.pos.tolist() == pos.tolist()
    edge_index, edge_attr = coalesce(data.edge_index, data.edge_attr, 2, 2)
    assert edge_index.tolist() == [[0, 0, 1], [0, 1, 0]]
    assert edge_attr.tolist() == [3, 1, 2]


def test_subgraph_convert():
    G = nx.complete_graph(5)

    edge_index = from_networkx(G).edge_index
    sub_edge_index_1, _ = subgraph([0, 1, 3, 4], edge_index)

    sub_edge_index_2 = from_networkx(G.subgraph([0, 1, 3, 4])).edge_index

    assert sub_edge_index_1.tolist() == sub_edge_index_2.tolist()


def test_vice_versa_convert():
    G = nx.complete_graph(5)
    assert G.is_directed() is False
    data = from_networkx(G)
    assert data.is_directed() is False
    G = to_networkx(data)
    assert G.is_directed() is True
    G = nx.to_undirected(G)
    assert G.is_directed() is False
