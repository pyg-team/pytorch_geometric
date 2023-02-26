import torch

from torch_geometric.data import Data
from torch_geometric.data.datapipes import DatasetAdapter
from torch_geometric.loader import DataLoader
from torch_geometric.testing import withPackage
from torch_geometric.utils import to_smiles


def create_dataset_adapter() -> DatasetAdapter:
    x = torch.randn(3, 8)
    edge_index = torch.tensor([[0, 1, 1], [1, 0, 2]])
    data = Data(x=x, edge_index=edge_index)
    return DatasetAdapter([data, data, data, data, data])


def test_dataset_adapter():
    dp = create_dataset_adapter()

    # Test DataLoader with the datapipe:
    loader = DataLoader(dp, batch_size=2, drop_last=True)
    batch = next(iter(loader))
    assert batch.x.shape == (6, 8)
    assert len(loader) == 2

    # Test apply sharding:
    dp.apply_sharding(2, 0)
    num_examples = sum(1 for _ in iter(dp))
    assert num_examples == 3


def test_datapipe_batch_graphs():
    dp = create_dataset_adapter()
    batch_dp = dp.batch_graphs(batch_size=2, drop_last=True)
    batch = next(iter(batch_dp))
    assert batch.x.shape == (6, 8)
    assert len(batch_dp) == 2


def test_functional_transform():
    dp = create_dataset_adapter()
    assert next(iter(dp)).is_directed()
    dp = dp.to_undirected()
    assert next(iter(dp)).is_undirected()


@withPackage('rdkit')
def test_datapipe_parse_smiles():
    smiles = 'F/C=C/F'
    dp = DatasetAdapter([smiles])
    dp = dp.parse_smiles()
    assert to_smiles(next(iter(dp))) == smiles

    dp = DatasetAdapter([{'abc': smiles, 'cba': '1.0'}])
    dp = dp.parse_smiles(smiles_key='abc', target_key='cba')
    assert to_smiles(next(iter(dp))) == smiles
