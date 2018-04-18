import torch
from torch_geometric.utils import add_self_loops, remove_self_loops


def test_add_self_loops():
    row = torch.LongTensor([0, 1, 0])
    col = torch.LongTensor([1, 0, 0])
    expected_output = [[0, 1, 0, 0, 1], [1, 0, 0, 0, 1]]

    output = add_self_loops(torch.stack([row, col], dim=0))
    assert output.tolist() == expected_output


def test_remove_self_loops():
    row = torch.LongTensor([1, 0, 1, 0, 2, 1])
    col = torch.LongTensor([0, 1, 1, 1, 2, 0])
    expected_output = [[1, 0, 0, 1], [0, 1, 1, 0]]

    output = remove_self_loops(torch.stack([row, col], dim=0))
    assert output.tolist() == expected_output
