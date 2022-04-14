from torch_geometric.testing import onlyFullTest


@onlyFullTest
def test_bzr(get_dataset):
    dataset = get_dataset(name='BZR')
    assert len(dataset) == 405
    assert dataset.num_features == 53
    assert dataset.num_node_labels == 53
    assert dataset.num_node_attributes == 0
    assert dataset.num_classes == 2
    assert str(dataset) == 'BZR(405)'
    assert len(dataset[0]) == 3


@onlyFullTest
def test_bzr_with_node_attr(get_dataset):
    dataset = get_dataset(name='BZR', use_node_attr=True)
    assert dataset.num_features == 56
    assert dataset.num_node_labels == 53
    assert dataset.num_node_attributes == 3
