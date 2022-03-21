from torch_geometric.datasets.elliptic import EllipticBitcoinDataset


def test_elliptic():
    dataset = EllipticBitcoinDataset('./')
    assert str(dataset) == 'EllipticBitcoinDataset()'
    assert len(dataset) == 1
    assert dataset.num_features == 165
    assert dataset.num_classes == 2
    assert len(dataset[0]) == 5
    assert dataset[0].edge_index.size() == (2, 234355)
    assert dataset[0].x.size() == (203769, 165)
    assert dataset[0].y.size() == (203769, )
    assert dataset[0].train_mask.size() == (203769, )
    assert dataset[0].train_mask.sum().item() == (3462, )
