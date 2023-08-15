import pytest
import torch

from torch_geometric.loader.utils import index_select


def test_index_select():
    x = torch.randn(3, 5)
    index = torch.tensor([0, 2])
    assert torch.equal(index_select(x, index), x[index])
    assert torch.equal(index_select(x, index, dim=-1), x[..., index])


def test_index_select_out_of_range():
    with pytest.raises(IndexError, match="out of range"):
        index_select(torch.randn(3, 5), torch.tensor([0, 2, 3]))
