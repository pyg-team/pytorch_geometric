import pytest
import torch

import torch_geometric
from torch_geometric.data.hypergraph_data import HyperGraphData
from torch_geometric.loader import DataLoader


def test_hypergraph_data():
    torch_geometric.set_debug(True)

    x = torch.tensor([[1, 3, 5, 7], [2, 4, 6, 8], [7, 8, 9, 10]],
                     dtype=torch.float).t()
    edge_index = torch.tensor([[0, 1, 2, 1, 2, 3, 0, 2, 3],
                               [0, 0, 0, 1, 1, 1, 2, 2, 2]])
    data = HyperGraphData(x=x, edge_index=edge_index).to(torch.device('cpu'))
    data.validate(raise_on_error=True)

    assert data.num_nodes == 4
    assert data.num_edges == 3

    assert data.node_attrs() == ['x']
    assert data.edge_attrs() == ['edge_index']

    assert data.x.tolist() == x.tolist()
    assert data['x'].tolist() == x.tolist()
    assert data.get('x').tolist() == x.tolist()
    assert data.get('y', 2) == 2
    assert data.get('y', None) is None

    assert sorted(data.keys()) == ['edge_index', 'x']
    assert len(data) == 2
    assert 'x' in data and 'edge_index' in data and 'pos' not in data

    D = data.to_dict()
    assert len(D) == 2
    assert 'x' in D and 'edge_index' in D

    D = data.to_namedtuple()
    assert len(D) == 2
    assert D.x is not None and D.edge_index is not None

    assert data.__cat_dim__('x', data.x) == 0
    assert data.__cat_dim__('edge_index', data.edge_index) == -1
    assert data.__inc__('x', data.x) == 0
    assert torch.equal(data.__inc__('edge_index', data.edge_index),
                       torch.tensor([[data.num_nodes], [data.num_edges]]))
    data_list = [data, data]
    loader = DataLoader(data_list, batch_size=2)
    batch = next(iter(loader))
    batched_edge_index = batch.edge_index
    assert batched_edge_index.tolist() == [[
        0, 1, 2, 1, 2, 3, 0, 2, 3, 4, 5, 6, 5, 6, 7, 4, 6, 7
    ], [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5]]

    assert not data.x.is_contiguous()
    data.contiguous()
    assert data.x.is_contiguous()

    assert not data.is_coalesced()
    data = data.coalesce()
    assert data.is_coalesced()

    clone = data.clone()
    assert clone != data
    assert len(clone) == len(data)
    assert clone.x.data_ptr() != data.x.data_ptr()
    assert clone.x.tolist() == data.x.tolist()
    assert clone.edge_index.data_ptr() != data.edge_index.data_ptr()
    assert clone.edge_index.tolist() == data.edge_index.tolist()

    data['x'] = x + 1
    assert data.x.tolist() == (x + 1).tolist()

    assert str(data) == 'HyperGraphData(x=[4, 3], edge_index=[2, 9])'

    dictionary = {'x': data.x, 'edge_index': data.edge_index}
    data = HyperGraphData.from_dict(dictionary)
    assert sorted(data.keys()) == ['edge_index', 'x']

    assert not data.has_isolated_nodes()
    # assert not data.has_self_loops()
    # assert data.is_undirected()
    # assert not data.is_directed()

    assert data.num_nodes == 4
    assert data.num_edges == 3
    with pytest.warns(UserWarning, match='deprecated'):
        assert data.num_faces is None
    assert data.num_node_features == 3
    assert data.num_features == 3

    data.edge_attr = torch.randn(data.num_edges, 2)
    assert data.num_edge_features == 2
    assert data.is_edge_attr('edge_attr')
    data.edge_attr = None

    data.x = None
    with pytest.warns(UserWarning, match='Unable to accurately infer'):
        assert data.num_nodes == 4

    data.edge_index = None
    with pytest.warns(UserWarning, match='Unable to accurately infer'):
        assert data.num_nodes is None
    assert data.num_edges == 0

    data.num_nodes = 4
    assert data.num_nodes == 4

    data = HyperGraphData(x=x, attribute=x)
    assert len(data) == 2
    assert data.x.tolist() == x.tolist()
    assert data.attribute.tolist() == x.tolist()

    face = torch.tensor([[0, 1], [1, 2], [2, 3]])
    data = HyperGraphData(num_nodes=4, face=face)
    with pytest.warns(UserWarning, match='deprecated'):
        assert data.num_faces == 2
    assert data.num_nodes == 4

    data = HyperGraphData(title='test')
    assert str(data) == "HyperGraphData(title='test')"
    assert data.num_node_features == 0
    # assert data.num_edge_features == 0

    key = value = 'test_value'
    data[key] = value
    assert data[key] == value
    del data[value]
    del data[value]  # Deleting unset attributes should work as well.

    assert data.get(key) is None
    assert data.get('title') == 'test'

    torch_geometric.set_debug(False)


def test_hypergraphdata_subgraph():
    x = torch.arange(5)
    y = torch.tensor([0.])
    edge_index = torch.tensor([[0, 1, 3, 2, 4, 0, 3, 4, 2, 1, 2, 3],
                               [0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3]])
    edge_attr = torch.rand(4, 2)
    data = HyperGraphData(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr,
                          num_nodes=5)

    out = data.subgraph(torch.tensor([1, 2, 4]))
    assert len(out) == 5
    assert torch.equal(out.x, torch.tensor([1, 2, 4]))
    assert torch.equal(out.y, data.y)
    assert out.edge_index.tolist() == [[1, 2, 2, 1, 0, 1], [0, 0, 1, 1, 2, 2]]
    assert torch.equal(out.edge_attr, edge_attr[[1, 2, 3]])
    assert out.num_nodes == 3

    # Test unordered selection:
    out = data.subgraph(torch.tensor([3, 1, 2]))
    assert len(out) == 5
    assert torch.equal(out.x, torch.tensor([3, 1, 2]))
    assert torch.equal(out.y, data.y)
    assert out.edge_index.tolist() == [[0, 2, 0, 2, 1, 2, 0],
                                       [0, 0, 1, 1, 2, 2, 2]]
    assert torch.equal(out.edge_attr, edge_attr[[1, 2, 3]])
    assert out.num_nodes == 3

    out = data.subgraph(torch.tensor([False, False, False, True, True]))
    assert len(out) == 5
    assert torch.equal(out.x, torch.arange(3, 5))
    assert torch.equal(out.y, data.y)
    assert out.edge_index.tolist() == [[0, 1, 0, 1], [0, 0, 1, 1]]
    assert torch.equal(out.edge_attr, edge_attr[[1, 2]])
    assert out.num_nodes == 2
