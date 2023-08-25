import pytest
import torch

from torch_geometric.data import Data
from torch_geometric.loader import CachedLoader, NeighborLoader
from torch_geometric.seed import seed_everything
from torch_geometric.testing import withCUDA, withPackage


@withCUDA
@withPackage('pyg_lib')
def test_cached_loader(device):
    seed_everything(12345)

    x = torch.randn(14, 16)
    edge_index = torch.tensor([
        [2, 3, 4, 5, 7, 7, 10, 11, 12, 13],
        [0, 1, 2, 3, 2, 3, 7, 7, 7, 7],
    ])
    data = Data(x=x, edge_index=edge_index)
    loader = NeighborLoader(
        data,
        num_neighbors=[2],
        batch_size=10,
        shuffle=False,
    )
    cached_loader = CachedLoader(loader, torch.device(device))
    edge_index_hook = cached_loader.attr_hooks[1]

    assert len(cached_loader) == len(loader)
    assert len(cached_loader.cache) == 0
    assert len(cached_loader.attr_hooks) == 3
    assert cached_loader.cache_filled is False
    assert cached_loader.hook_reg_open is True

    for idx, batch in enumerate(cached_loader):
        assert len(cached_loader.cache) == (idx + 1)
        assert batch is cached_loader.cache[idx]
        # check if batch consist of default attributes
        assert len(batch) == len(cached_loader.attr_hooks)
        n_id, edge_index, bs = batch
        assert isinstance(n_id, torch.Tensor)
        assert isinstance(edge_index, torch.Tensor)
        assert isinstance(bs, int)

    assert len(cached_loader.cache) == len(loader)
    assert cached_loader.cache_filled is True
    assert cached_loader.hook_reg_open is False

    cached_loader = CachedLoader(loader, torch.device(device))
    cached_loader.reset_attr_hooks()

    assert len(cached_loader.attr_hooks) == 0

    cached_loader.register_attr_hooks([edge_index_hook])
    edge_index_orig = edge_index_hook(next(iter(loader))).to(device)
    edge_index_cached = next(iter(cached_loader))[0]

    assert torch.all(edge_index_orig == edge_index_cached)
    with pytest.raises(RuntimeError):
        cached_loader.reset_attr_hooks()
