import torch
from torch_geometric.utils import one_hot


def test_one_hot():
    src = torch.LongTensor([1, 0, 3])
    expected_output = [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]]

    assert one_hot(src).tolist() == expected_output
    assert one_hot(src, 4).tolist() == expected_output

    src = torch.LongTensor([[1, 0], [0, 1], [2, 0]])
    expected_output = [[0, 1, 0, 1, 0], [1, 0, 0, 0, 1], [0, 0, 1, 1, 0]]
    assert one_hot(src).tolist() == expected_output
    assert one_hot(src, torch.tensor([3, 2])).tolist() == expected_output
