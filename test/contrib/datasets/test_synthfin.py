import pytest

from torch_geometric.contrib.datasets import SynthFinDataset
from torch_geometric.testing import onlyFullTest, onlyOnline


@pytest.mark.dataset
@onlyOnline
@onlyFullTest
def test_synthfin_dataset(tmp_path):
    dataset = SynthFinDataset(root=str(tmp_path))

    assert len(dataset) == 1
    assert dataset.__class__.__name__ == 'SynthFinDataset'
    assert dataset.url.endswith('raw.zip')

    data = dataset[0]
    assert data.num_nodes == 100000
    assert data.num_features == 10
    assert data.y.size(0) == 100000
