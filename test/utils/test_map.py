import torch

from torch_geometric.utils.map import map_index


def test_map_index():
    src = torch.tensor([2, 0, 1, 0, 3])
    index = torch.tensor([3, 2, 0, 1])

    out, mask = map_index(src, index)
    assert out.tolist() == [1, 2, 3, 2, 0]
    assert mask.tolist() == [True, True, True, True, True]


def test_map_index_na():
    src = torch.tensor([2, 0, 1, 0, 3])
    index = torch.tensor([3, 2, 0])

    out, mask = map_index(src, index)
    assert out.tolist() == [1, 2, 2, 0]
    assert mask.tolist() == [True, True, False, True, True]
