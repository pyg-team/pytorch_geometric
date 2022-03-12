from torch_geometric.testing import withDataset


@withDataset(name='MUTAG')
def test_mutag(get_dataset):
    dataset = get_dataset()
    assert len(dataset) == 188
    assert dataset.num_features == 7
    assert dataset.num_classes == 2
    assert str(dataset) == 'MUTAG(188)'

    assert len(dataset[0]) == 4
    assert dataset[0].edge_attr.size(1) == 4


@withDataset(name='MUTAG')
def test_mutag_with_node_attr(get_dataset):
    dataset = get_dataset(use_node_attr=True)
    assert dataset.num_features == 7
