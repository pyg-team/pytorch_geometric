import pytest

from torch_geometric.datasets import WebQSPDataset
from torch_geometric.testing import onlyFullTest, onlyOnline


@pytest.mark.skip(reason="Times out")
@onlyOnline
@onlyFullTest
def test_web_qsp_dataset(tmp_path):
    dataset = WebQSPDataset(root=tmp_path)
    assert len(dataset) == 4700
    assert str(dataset) == "WebQSPDataset(4700)"


@onlyOnline
@onlyFullTest
def test_web_qsp_dataset_limit(tmp_path):
    dataset = WebQSPDataset(root=tmp_path, limit=100)
    assert len(dataset) == 100
    assert str(dataset) == "WebQSPDataset(100)"


@onlyOnline
@onlyFullTest
def test_web_qsp_dataset_multiple_splits(tmp_path):
    dataset = WebQSPDataset(root=tmp_path, split='train')
    assert str(
        dataset) == f"WebQSPDataset({len(dataset.split_indices['train'])})"
    dataset = WebQSPDataset(root=tmp_path, split='val')
    assert str(
        dataset) == f"WebQSPDataset({len(dataset.split_indices['val'])})"


@onlyOnline
@onlyFullTest
def test_web_qsp_dataset_limit_no_pcst(tmp_path):
    dataset = WebQSPDataset(root=tmp_path, limit=100, include_pcst=False)
    assert len(dataset) == 100
    assert str(dataset) == "WebQSPDataset(100)"