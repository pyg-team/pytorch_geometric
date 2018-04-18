import torch
from torch_geometric.utils import coalesce


def test_coalesce():
    row = torch.LongTensor([1, 0, 1, 0, 2, 1])
    col = torch.LongTensor([0, 1, 1, 1, 0, 0])
    expected_output = [[0, 1, 1, 2], [1, 0, 1, 0]]

    output = coalesce(torch.stack([row, col], dim=0))
    assert output.tolist() == expected_output
