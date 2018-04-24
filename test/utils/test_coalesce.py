import torch
from torch_geometric.utils import coalesce


def test_coalesce():
    row = torch.LongTensor([1, 0, 1, 0, 2, 1])
    col = torch.LongTensor([0, 1, 1, 1, 0, 0])
    expected_output = [[0, 1, 1, 2], [1, 0, 1, 0]]
    expected_perm = [1, 0, 2, 4]

    output, output_perm = coalesce(torch.stack([row, col], dim=0))
    assert output.tolist() == expected_output
    assert output_perm.tolist() == expected_perm
