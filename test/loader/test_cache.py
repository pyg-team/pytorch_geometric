import torch
from torch import Tensor

from torch_geometric.data import Data
from torch_geometric.loader import CachedLoader, NeighborLoader
from torch_geometric.testing import withCUDA, withPackage


@withCUDA
@withPackage('pyg_lib')
def test_cached_loader(device):
    x = torch.randn(14, 16)
    edge_index = torch.tensor([
        [2, 3, 4, 5, 7, 7, 10, 11, 12, 13],
        [0, 1, 2, 3, 2, 3, 7, 7, 7, 7],
    ])

    loader = NeighborLoader(
        Data(x=x, edge_index=edge_index),
        num_neighbors=[2],
        batch_size=10,
        shuffle=False,
    )
    cached_loader = CachedLoader(loader, device=device)

    assert len(cached_loader) == len(loader)
    assert len(cached_loader._cache) == 0

    cache = []
    for i, batch in enumerate(cached_loader):
        assert len(cached_loader._cache) == i + 1
        assert batch.x.device == device
        assert batch.edge_index.device == device

        cache.append(batch)

    for i, batch in enumerate(cached_loader):
        assert batch == cache[i]

    cached_loader.clear()
    assert len(cached_loader._cache) == 0


@withCUDA
@withPackage('pyg_lib')
def test_cached_loader_transform(device):
    x = torch.randn(14, 16)
    edge_index = torch.tensor([
        [2, 3, 4, 5, 7, 7, 10, 11, 12, 13],
        [0, 1, 2, 3, 2, 3, 7, 7, 7, 7],
    ])

    loader = NeighborLoader(
        Data(x=x, edge_index=edge_index),
        num_neighbors=[2],
        batch_size=10,
        shuffle=False,
    )
    cached_loader = CachedLoader(
        loader,
        device=device,
        transform=lambda batch: batch.edge_index,
    )

    assert len(cached_loader) == len(loader)
    assert len(cached_loader._cache) == 0

    cache = []
    for i, batch in enumerate(cached_loader):
        assert len(cached_loader._cache) == i + 1
        assert isinstance(batch, Tensor)
        assert batch.dim() == 2 and batch.size(0) == 2
        assert batch.device == device

        cache.append(batch)

    for i, batch in enumerate(cached_loader):
        assert torch.equal(batch, cache[i])
