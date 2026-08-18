import pytest

from torch_geometric.testing import withPackage


@pytest.mark.dataset
@withPackage('h5py')
def test_oscillator_dataset(get_dataset):
    dataset = get_dataset(name='osc20', split='train')
    assert len(dataset) == 7000
    assert 'OscillatorDataset' in str(dataset)

    data = dataset[0]
    assert data.num_nodes == 20
    assert data.num_edges == 54
    assert data.x.size(1) > 0
    assert data.y.size(0) == data.num_nodes


@pytest.mark.dataset
@withPackage('h5py')
def test_oscillator_dataset_texas(get_dataset):
    dataset = get_dataset(name='osctexas')
    assert len(dataset) == 1
    assert 'OscillatorDataset' in str(dataset)

    data = dataset[0]
    assert data.num_nodes == 1910
    assert data.x.size(1) > 0
    assert data.edge_index.size(1) == 5154
    assert data.y.size(0) == data.num_nodes
