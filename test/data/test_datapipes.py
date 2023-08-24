import pytest
import torch

from torch_geometric.data import Data
from torch_geometric.data.datapipes import DatasetAdapter
from torch_geometric.loader import DataLoader
from torch_geometric.testing import withPackage
from torch_geometric.utils import to_smiles


@pytest.fixture()
def dataset_adapter() -> DatasetAdapter:
    x = torch.randn(3, 8)
    edge_index = torch.tensor([[0, 1, 1], [1, 0, 2]])
    data = Data(x=x, edge_index=edge_index)
    return DatasetAdapter([data, data, data, data])


def test_dataset_adapter(dataset_adapter):
    loader = DataLoader(dataset_adapter, batch_size=2)
    batch = next(iter(loader))
    assert batch.x.shape == (6, 8)
    assert len(loader) == 2

    # Test sharding:
    dataset_adapter.apply_sharding(2, 0)
    assert len([data for data in dataset_adapter]) == 2

    assert dataset_adapter.is_shardable()


def test_datapipe_batch_graphs(dataset_adapter):
    dp = dataset_adapter.batch_graphs(batch_size=2)
    assert len(dp) == 2
    batch = next(iter(dp))
    assert batch.x.shape == (6, 8)


def test_functional_transform(dataset_adapter):
    assert next(iter(dataset_adapter)).is_directed()
    dataset_adapter = dataset_adapter.to_undirected()
    assert next(iter(dataset_adapter)).is_undirected()


@withPackage('rdkit')
def test_datapipe_parse_smiles():
    smiles = 'F/C=C/F'

    dp = DatasetAdapter([smiles])
    dp = dp.parse_smiles()
    assert to_smiles(next(iter(dp))) == smiles

    dp = DatasetAdapter([{'abc': smiles, 'cba': '1.0'}])
    dp = dp.parse_smiles(smiles_key='abc', target_key='cba')
    assert to_smiles(next(iter(dp))) == smiles
