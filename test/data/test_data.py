import torch
from torch_geometric.data import Data


def test_data():
    x = torch.tensor([[1, 3, 5], [2, 4, 6]], dtype=torch.float).t()
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    data = Data(x=x, edge_index=edge_index)

    assert data.x.tolist() == x.tolist()
    assert data['x'].tolist() == x.tolist()

    assert sorted(data.keys) == ['edge_index', 'x']
    assert len(data) == 2
    assert 'x' in data and 'edge_index' in data and 'pos' not in data

    assert data.cat_dim('x') == 0
    assert data.cat_dim('edge_index') == -1

    data.to(torch.device('cpu'))
    assert not data.x.is_contiguous()
    data.contiguous()
    assert data.x.is_contiguous()

    data['x'] = x + 1
    assert data.x.tolist() == (x + 1).tolist()

    assert data.__repr__() == 'Data(edge_index=[2, 4], x=[3, 2])'

    dictionary = {'x': x, 'edge_index': edge_index}
    data = Data.from_dict(dictionary)
    assert sorted(data.keys) == ['edge_index', 'x']

    assert not data.contains_isolated_nodes()
    assert not data.contains_self_loops()
    assert data.is_undirected()
    assert not data.is_directed()

    assert data.num_nodes == 3
    assert data.num_edges == 4

    data.x = None
    assert data.num_nodes == 3

    data.edge_index = None
    assert data.num_nodes is None
    assert data.num_edges is None
