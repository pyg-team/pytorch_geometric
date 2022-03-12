from torch_geometric.testing import withDataset


@withDataset(name='BZR')
def test_bzr(dataset):
    assert len(dataset) == 405
    assert dataset.num_features == 53
    assert dataset.num_node_labels == 53
    assert dataset.num_node_attributes == 0
    assert dataset.num_classes == 2
    assert str(dataset) == 'BZR(405)'
    assert len(dataset[0]) == 3


@withDataset(name='BZR', use_node_attr=True)
def test_bzr_with_node_attr(dataset):
    assert dataset.num_features == 56
    assert dataset.num_node_labels == 53
    assert dataset.num_node_attributes == 3
