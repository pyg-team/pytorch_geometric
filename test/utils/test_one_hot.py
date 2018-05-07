import torch
from torch_geometric.utils import one_hot


def test_one_hot():
    src = torch.LongTensor([1, 0, 3])
    expected_output = [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]]

    output = one_hot(src)
    assert output.tolist() == expected_output
