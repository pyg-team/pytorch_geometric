import torch
from torch_geometric.data import Data
from torch_sparse import coalesce


def test_data():
    x = torch.tensor([[1, 3, 5], [2, 4, 6]], dtype=torch.float).t()
    edge_index = torch.tensor([[0, 0, 1, 1, 2], [1, 1, 0, 2, 1]])
    data = Data(x=x, edge_index=edge_index).to(torch.device('cpu'))

    N = data.num_nodes

    assert data.x.tolist() == x.tolist()
    assert data['x'].tolist() == x.tolist()

    assert sorted(data.keys) == ['edge_index', 'x']
    assert len(data) == 2
    assert 'x' in data and 'edge_index' in data and 'pos' not in data

    assert data.__cat_dim__('x', data.x) == 0
    assert data.__cat_dim__('edge_index', data.edge_index) == -1
    assert data.__cumsum__('x', data.x) is False
    assert data.__cumsum__('edge_index', data.edge_index) is True

    assert not data.x.is_contiguous()
    data.contiguous()
    assert data.x.is_contiguous()

    assert not data.is_coalesced()
    data.edge_index, _ = coalesce(data.edge_index, None, N, N)
    assert data.is_coalesced()

    clone = data.clone()
    assert clone != data
    assert len(clone) == len(data)
    assert clone.x.tolist() == data.x.tolist()
    assert clone.edge_index.tolist() == data.edge_index.tolist()

    data['x'] = x + 1
    assert data.x.tolist() == (x + 1).tolist()

    assert data.__repr__() == 'Data(edge_index=[2, 4], x=[3, 2])'

    dictionary = {'x': data.x, 'edge_index': data.edge_index}
    data = Data.from_dict(dictionary)
    assert sorted(data.keys) == ['edge_index', 'x']

    assert not data.contains_isolated_nodes()
    assert not data.contains_self_loops()
    assert data.is_undirected()
    assert not data.is_directed()

    assert data.num_nodes == 3
    assert data.num_edges == 4
    assert data.num_faces is None
    assert data.num_features == 2

    data.x = None
    assert data.num_nodes == 3

    data.edge_index = None
    assert data.num_nodes is None
    assert data.num_edges is None

    data.num_nodes = 4
    assert data.num_nodes == 4

    data = Data(x=x, attribute=x)
    assert len(data) == 2
    assert data.x.tolist() == x.tolist()
    assert data.attribute.tolist() == x.tolist()

    face = torch.tensor([[0, 1], [1, 2], [2, 3]])
    data = Data(face=face)
    assert data.num_faces == 2
