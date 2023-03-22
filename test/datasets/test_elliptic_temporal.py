from torch_geometric.testing import onlyFullTest


@onlyFullTest
def test_elliptic_bitcoin_dataset(get_dataset):
    dataset = get_dataset(name='EllipticBitcoinTemporalDataset')
    assert str(dataset) == 'EllipticBitcoinDataset()'
    assert len(dataset) == 1
    assert dataset.num_features == 165
    assert dataset.num_classes == 2

    data = dataset[0]
    assert len(data) == 6
    assert data.x.size() == (7880, 165)
    assert data.edge_index.size() == (2, 9164)
    assert data.y.size() == (7880, )

    assert data.train_mask.size() == (7880, )
    assert data.train_mask.sum() > 0
    assert data.test_mask.size() == (7880, )
    assert data.test_mask.sum() == 0
    assert data.train_mask.sum() + data.test_mask.sum() == 2147 + 0
    assert data.y[data.train_mask].sum() == 17
    assert data.y[data.test_mask].sum() == 0
    assert data.y[data.train_mask].sum() + data.y[data.test_mask].sum() == 17
    assert data.y[data.test_mask | data.train_mask].min() == 0
    assert data.y[data.test_mask | data.train_mask].max() == 1
