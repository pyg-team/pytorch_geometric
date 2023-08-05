import pytest
import torch

from torch_geometric.loader.utils import index_select


@pytest.mark.parametrize("", [1, 2, 3])
def test_index_select():
    x = torch.randn(3, 5)
    index = torch.tensor([0, 2])
    assert torch.all(index_select(x, index) == x[index])
    assert torch.all(index_select(x, index, dim=-1) == x[..., index])


def test_index_select_invalid_value():
    with pytest.raises(ValueError, "Encountered invalid feature tensor"):
        index_select(torch.randn(3, 5), torch.tensor([0, 2, 3]))
