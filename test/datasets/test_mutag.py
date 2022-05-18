def test_mutag(get_dataset):
    dataset = get_dataset(name='MUTAG')
    assert len(dataset) == 188
    assert dataset.num_features == 7
    assert dataset.num_classes == 2
    assert str(dataset) == 'MUTAG(188)'

    assert len(dataset[0]) == 4
    assert dataset[0].edge_attr.size(1) == 4


def test_mutag_with_node_attr(get_dataset):
    dataset = get_dataset(name='MUTAG', use_node_attr=True)
    assert dataset.num_features == 7
