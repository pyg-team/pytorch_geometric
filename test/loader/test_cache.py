import torch

from torch_geometric.data import Data
from torch_geometric.loader import CachedLoader, NeighborLoader


def test_cached_loader():
    zeros = torch.zeros(10, dtype=torch.long)
    ones = torch.ones(90, dtype=torch.long)

    y = torch.cat([zeros, ones], dim=0)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    data = Data(edge_index=edge_index, y=y, num_nodes=y.size(0))

    loader = NeighborLoader(data, batch_size=10, num_neighbors=[-1])
    cached_loader = CachedLoader(loader, torch.device('cpu'))

    assert len(cached_loader) == len(loader)
    assert len(cached_loader.cache) == 0
    assert cached_loader.cache_filled is False

    for idx, batch in enumerate(cached_loader):
        assert len(cached_loader.cache) == (idx + 1)
        assert batch is cached_loader.cache[idx]

    assert len(cached_loader.cache) == len(loader)
    assert cached_loader.cache_filled is True
