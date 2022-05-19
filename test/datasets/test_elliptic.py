from torch_geometric.testing import onlyFullTest


@onlyFullTest
def test_elliptic_bitcoin_dataset(get_dataset):
    dataset = get_dataset(name='EllipticBitcoinDataset')
    assert str(dataset) == 'EllipticBitcoinDataset()'
    assert len(dataset) == 1
    assert dataset.num_features == 165
    assert dataset.num_classes == 2

    data = dataset[0]
    assert len(data) == 5
    assert data.x.size() == (203769, 165)
    assert data.edge_index.size() == (2, 234355)
    assert data.y.size() == (203769, )

    assert data.train_mask.size() == (203769, )
    assert data.train_mask.sum() > 0
    assert data.test_mask.size() == (203769, )
    assert data.test_mask.sum() > 0
    assert data.train_mask.sum() + data.test_mask.sum() == 4545 + 42019
    assert data.y[data.train_mask].sum() == 3462
    assert data.y[data.test_mask].sum() == 1083
    assert data.y[data.train_mask].sum() + data.y[data.test_mask].sum() == 4545
    assert data.y[data.test_mask | data.train_mask].min() == 0
    assert data.y[data.test_mask | data.train_mask].max() == 1
