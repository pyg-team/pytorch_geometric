import pytest
import torch

from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader, ZipLoader


@pytest.mark.parametrize('filter_per_worker', [True, False])
def test_zip_loader(filter_per_worker):
    x = torch.arange(100)
    edge_index = torch.randint(0, 100, (2, 1000))
    data = Data(x=x, edge_index=edge_index)

    loaders = [
        NeighborLoader(data, [5], input_nodes=torch.arange(0, 50)),
        NeighborLoader(data, [5], input_nodes=torch.arange(50, 95)),
    ]

    loader = ZipLoader(loaders, batch_size=10,
                       filter_per_worker=filter_per_worker)

    assert str(loader) == ('ZipLoader(loaders=[NeighborLoader(), '
                           'NeighborLoader()])')
    assert len(loader) == 5
    assert loader.dataset == range(0, 45)

    for i, (batch1, batch2) in enumerate(loader):
        n_id1 = batch1.n_id[:batch1.batch_size]
        n_id2 = batch2.n_id[:batch2.batch_size]

        if i < 4:
            assert batch1.batch_size == 10
            assert batch2.batch_size == 10
            assert torch.equal(n_id1, torch.arange(0 + i * 10, 10 + i * 10))
            assert torch.equal(n_id2, torch.arange(50 + i * 10, 60 + i * 10))
        else:
            assert batch1.batch_size == 5
            assert batch2.batch_size == 5
            assert torch.equal(n_id1, torch.arange(0 + i * 10, 5 + i * 10))
            assert torch.equal(n_id2, torch.arange(50 + i * 10, 55 + i * 10))
