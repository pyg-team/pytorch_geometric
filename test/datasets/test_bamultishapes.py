import pytest


def test_bamultishapes(get_dataset):
    dataset = get_dataset(name='BAMultiShapesDataset')

    assert str(dataset) == 'BAMultiShapesDataset(1000)'
    assert len(dataset) == 1000
    assert dataset.num_features == 10
    assert dataset.num_classes == 2

    data = dataset[0]
    assert len(data) == 3
    assert data.edge_index.size(1) >= 45
    assert data.x.size() == (40, 10)
    assert data.y.size() == (1, )
