import torch
from torch_geometric.utils import add_self_loops, remove_self_loops


def test_add_self_loops():
    edge_index = torch.LongTensor([[0, 1, 0], [1, 0, 0]])
    expected_output = [[0, 1, 0, 0, 1], [1, 0, 0, 0, 1]]

    output = add_self_loops(edge_index)
    assert output.tolist() == expected_output


def test_remove_self_loops():
    edge_index = torch.LongTensor([[1, 0, 1, 0, 2, 1], [0, 1, 1, 1, 2, 0]])
    expected_output = [[1, 0, 0, 1], [0, 1, 1, 0]]

    output = remove_self_loops(edge_index)
    assert output.tolist() == expected_output
