import torch

from torch_geometric.utils import index_to_mask


def test_index_to_mask():
    index = torch.tensor([1, 3, 5])

    mask = index_to_mask(index)
    assert mask.tolist() == [False, True, False, True, False, True]

    mask = index_to_mask(index, size=7)
    assert mask.tolist() == [False, True, False, True, False, True, False]
