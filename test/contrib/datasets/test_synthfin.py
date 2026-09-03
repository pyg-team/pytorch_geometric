import pytest

from torch_geometric.contrib.datasets import SynthFinDataset
from torch_geometric.testing import onlyFullTest, onlyOnline


@pytest.mark.dataset
@onlyOnline
@onlyFullTest
def test_synthfin_dataset(tmp_path):
    dataset = SynthFinDataset(root=str(tmp_path))

    assert len(dataset) == 3
    assert dataset.__class__.__name__ == 'SynthFinDataset'
    assert dataset.url.endswith('raw.zip')

    for i in range(3):
        data = dataset[i]
        assert data.num_nodes == 100000
        assert data.num_features == 10
        assert data.y.size(0) == 100000

        if i == 0:
            assert data.edge_time.max().item() <= 7
            assert data.train_mask.sum().item() > 0
            assert data.val_mask.sum().item() == 0
            assert data.test_mask.sum().item() == 0
        elif i == 1:
            assert data.edge_time.max().item() <= 8
            assert data.train_mask.sum().item() == 0
            assert data.val_mask.sum().item() > 0
            assert data.test_mask.sum().item() == 0
        else:
            assert data.edge_time.max().item() <= 10
            assert data.train_mask.sum().item() == 0
            assert data.val_mask.sum().item() == 0
            assert data.test_mask.sum().item() > 0
