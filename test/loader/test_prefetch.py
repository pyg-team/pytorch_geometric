import torch

from torch_geometric.loader.prefetch import GPUPrefetcher
from torch_geometric.testing import onlyCUDA


@onlyCUDA
def test_gpu_prefetcher():
    data = [torch.randn(5, 5) for _ in range(10)]

    loader = GPUPrefetcher(data, device='cuda')
    assert str(loader).startswith('GPUPrefetcher')
    assert len(loader) == 10

    for i, batch in enumerate(loader):
        assert batch.is_cuda
        assert torch.equal(batch.cpu(), data[i])
        assert loader.idx > 0
    assert loader.idx == 0
