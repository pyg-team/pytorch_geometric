import torch
from torch_geometric.utils import softmax


def test_softmax():
    src = torch.tensor([1., 1., 1., 1.])
    index = torch.tensor([0, 0, 1, 2])
    ptr = torch.tensor([0, 2, 3, 4])

    out = softmax(src, index)
    assert out.tolist() == [0.5, 0.5, 1, 1]
    assert softmax(src, None, ptr).tolist() == out.tolist()
