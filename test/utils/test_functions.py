from math import prod

import pytest
import torch

from torch_geometric.testing import withDevice
from torch_geometric.utils import cumsum


def test_cumsum():
    x = torch.tensor([2, 4, 1])
    assert cumsum(x).tolist() == [0, 2, 6, 7]

    x = torch.tensor([[2, 4], [3, 6]])
    assert cumsum(x, dim=1).tolist() == [[0, 2, 6], [0, 3, 9]]


@withDevice
@pytest.mark.parametrize('shape', [(3, ), (2, 3), (2, 3, 4), (0, ), (2, 0, 3)])
@pytest.mark.parametrize('dtype', [torch.float, torch.long])
@pytest.mark.parametrize('negative', [False, True])
def test_cumsum_dim(shape, dtype, negative, device):
    x = torch.arange(prod(shape), dtype=dtype, device=device).view(shape)

    for dim in range(x.dim()):
        size = list(x.size())
        size[dim] = 1
        expected = torch.cat([x.new_zeros(size), x.cumsum(dim)], dim=dim)
        out = cumsum(x, dim=dim - x.dim() if negative else dim)
        assert torch.equal(out, expected)


@withDevice
@pytest.mark.parametrize('dim', [-3, -2, -1, 0, 1, 2])
def test_cumsum_non_contiguous(dim, device):
    x = torch.arange(24, device=device).view(2, 3, 4).transpose(0, 2)
    assert not x.is_contiguous()
    size = list(x.size())
    size[dim] = 1
    expected = torch.cat([x.new_zeros(size), x.cumsum(dim)], dim=dim)
    assert torch.equal(cumsum(x, dim=dim), expected)


@pytest.mark.parametrize('dim', [-4, 3])
def test_cumsum_invalid_dim(dim):
    with pytest.raises(IndexError, match="Dimension out of range"):
        cumsum(torch.ones(2, 3, 4), dim=dim)
