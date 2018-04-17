import torch
from torch_geometric.utils import coalesce


def test_coalesce():
    edge_index = torch.LongTensor([[1, 0, 1, 0, 2, 1], [0, 1, 1, 1, 0, 0]])
    expected_output = [[0, 1, 1, 2], [1, 0, 1, 0]]

    output = coalesce(edge_index)
    assert output.tolist() == expected_output
