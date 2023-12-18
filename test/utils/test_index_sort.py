import torch

from torch_geometric.testing import withCUDA
from torch_geometric.utils import index_sort


@withCUDA
def test_index_sort_stable(device):
    for _ in range(100):
        inputs = torch.randint(0, 4, size=(10, ), device=device)

        out = index_sort(inputs, stable=True)
        expected = torch.sort(inputs, stable=True)

        assert torch.equal(out[0], expected[0])
        assert torch.equal(out[1], expected[1])
