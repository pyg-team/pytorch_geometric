from torch_geometric.datasets import BAShapes


def test_ba_shapes():
    dataset = BAShapes(seed=0)

    assert len(dataset) == 1
    assert dataset.num_features == 10
    assert dataset.num_classes == 4
    assert dataset.__repr__() == 'BAShapes()'

    assert len(dataset[0]) == 5
    assert dataset[0].edge_index.size() == (2, 4070)
    assert dataset[0].x.size() == (700, 10)
    assert dataset[0].y.size() == (700, )
    assert sum(dataset[0].test_mask) == (60, )
    assert sum(dataset[0].edge_label) == (960)
