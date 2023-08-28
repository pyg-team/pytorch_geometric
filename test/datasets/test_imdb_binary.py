from torch_geometric.testing import onlyFullTest, onlyOnline


@onlyOnline
@onlyFullTest
def test_imdb_binary(get_dataset):
    dataset = get_dataset(name='IMDB-BINARY')
    assert len(dataset) == 1000
    assert dataset.num_features == 0
    assert dataset.num_classes == 2
    assert str(dataset) == 'IMDB-BINARY(1000)'

    data = dataset[0]
    assert len(data) == 3
    assert data.edge_index.size() == (2, 146)
    assert data.y.size() == (1, )
    assert data.num_nodes == 20
