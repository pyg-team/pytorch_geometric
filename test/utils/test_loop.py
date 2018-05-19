import torch
from torch_geometric.utils import (contains_self_loops, add_self_loops,
                                   remove_self_loops)


def test_contains_self_loops():
    row = torch.tensor([0, 1, 0])
    col = torch.tensor([1, 0, 0])

    assert contains_self_loops(torch.stack([row, col], dim=0)) is True

    row = torch.tensor([0, 1, 1])
    col = torch.tensor([1, 0, 2])

    assert contains_self_loops(torch.stack([row, col], dim=0)) is False


def test_add_self_loops():
    row = torch.tensor([0, 1, 0])
    col = torch.tensor([1, 0, 0])
    expected_output = [[0, 1, 0, 0, 1], [1, 0, 0, 0, 1]]

    output = add_self_loops(torch.stack([row, col], dim=0))
    assert output.tolist() == expected_output


def test_remove_self_loops():
    row = torch.tensor([1, 0, 1, 0, 2, 1])
    col = torch.tensor([0, 1, 1, 1, 2, 0])
    expected_output = [[1, 0, 0, 1], [0, 1, 1, 0]]

    output = remove_self_loops(torch.stack([row, col], dim=0))
    assert output.tolist() == expected_output
