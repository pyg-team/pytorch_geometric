import pytest

from torch_geometric.datasets import FakeDataset, FakeHeteroDataset


@pytest.mark.parametrize('num_graphs', [1, 10])
@pytest.mark.parametrize('edge_dim', [0, 1, 4])
@pytest.mark.parametrize('task', ['node', 'graph', 'auto'])
def test_fake_dataset(num_graphs, edge_dim, task):
    dataset = FakeDataset(num_graphs, edge_dim=edge_dim, task=task,
                          global_features=3)

    if num_graphs > 1:
        assert str(dataset) == f'FakeDataset({num_graphs})'
    else:
        assert str(dataset) == 'FakeDataset()'

    assert len(dataset) == num_graphs

    data = dataset[0]

    assert data.num_features == 64

    if edge_dim == 0:
        assert len(data) == 4
    elif edge_dim == 1:
        assert len(data) == 5
        assert data.edge_weight.size() == (data.num_edges, )
        assert data.edge_weight.min() >= 0 and data.edge_weight.max() < 1
    else:
        assert len(data) == 5
        assert data.edge_attr.size() == (data.num_edges, edge_dim)
        assert data.edge_attr.min() >= 0 and data.edge_attr.max() < 1

    assert data.y.min() >= 0 and data.y.max() < 10
    if task == 'node' or (task == 'auto' and num_graphs == 1):
        assert data.y.size() == (data.num_nodes, )
    else:
        assert data.y.size() == (1, )

    assert data.global_features.size() == (3, )


@pytest.mark.parametrize('num_graphs', [1, 10])
@pytest.mark.parametrize('edge_dim', [0, 1, 4])
@pytest.mark.parametrize('task', ['node', 'graph', 'auto'])
def test_fake_hetero_dataset(num_graphs, edge_dim, task):
    dataset = FakeHeteroDataset(num_graphs, edge_dim=edge_dim, task=task,
                                global_features=3)

    if num_graphs > 1:
        assert str(dataset) == f'FakeHeteroDataset({num_graphs})'
    else:
        assert str(dataset) == 'FakeHeteroDataset()'

    assert len(dataset) == num_graphs

    data = dataset[0]

    for store in data.node_stores:
        assert store.num_features > 0

        if task == 'node' or (task == 'auto' and num_graphs == 1):
            if store._key == 'v0':
                assert store.y.min() >= 0 and store.y.max() < 10
                assert store.y.size() == (store.num_nodes, )

    for store in data.edge_stores:
        if edge_dim == 0:
            assert len(data) == 4
        elif edge_dim == 1:
            assert len(data) == 5
            assert store.edge_weight.size() == (store.num_edges, )
            assert store.edge_weight.min() >= 0 and store.edge_weight.max() < 1
        else:
            assert len(data) == 5
            assert store.edge_attr.size() == (store.num_edges, edge_dim)
            assert store.edge_attr.min() >= 0 and store.edge_attr.max() < 1

    if task == 'graph' or (task == 'auto' and num_graphs > 1):
        assert data.y.min() >= 0 and data.y.max() < 10
        assert data.y.size() == (1, )

    assert data.global_features.size() == (3, )
