import torch

from torch_geometric.utils import one_hot


def test_one_hot():
    index = torch.tensor([0, 1, 2])

    out = one_hot(index)
    assert out.size() == (3, 3)
    assert out.dtype == torch.float
    assert out.tolist() == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    out = one_hot(index, num_classes=4, dtype=torch.long)
    assert out.size() == (3, 4)
    assert out.dtype == torch.long
    assert out.tolist() == [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
