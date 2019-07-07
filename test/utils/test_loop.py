import torch
from torch_geometric.utils import (contains_self_loops, remove_self_loops,
                                   segregate_self_loops, add_self_loops,
                                   add_remaining_self_loops)


def test_contains_self_loops():
    edge_index = torch.tensor([[0, 1, 0], [1, 0, 0]])
    assert contains_self_loops(edge_index)

    edge_index = torch.tensor([[0, 1, 1], [1, 0, 2]])
    assert not contains_self_loops(edge_index)


def test_remove_self_loops():
    edge_index = torch.tensor([[0, 1, 0], [1, 0, 0]])
    edge_attr = [[1, 2], [3, 4], [5, 6]]
    edge_attr = torch.Tensor(edge_attr)

    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    assert edge_index.tolist() == [[0, 1], [1, 0]]
    assert edge_attr.tolist() == [[1, 2], [3, 4]]


def test_segregate_self_loops():
    edge_index = torch.tensor([[0, 0, 1], [0, 1, 0]])

    out = segregate_self_loops(edge_index)
    assert out[0].tolist() == [[0, 1], [1, 0]]
    assert out[1] is None
    assert out[2].tolist() == [[0], [0]]
    assert out[3] is None

    edge_attr = torch.tensor([1, 2, 3])
    out = segregate_self_loops(edge_index, edge_attr)
    assert out[0].tolist() == [[0, 1], [1, 0]]
    assert out[1].tolist() == [2, 3]
    assert out[2].tolist() == [[0], [0]]
    assert out[3].tolist() == [1]


def test_add_self_loops():
    edge_index = torch.tensor([[0, 1, 0], [1, 0, 0]])

    expected = [[0, 1, 0, 0, 1], [1, 0, 0, 0, 1]]
    assert add_self_loops(edge_index)[0].tolist() == expected

    edge_weight = torch.tensor([0.5, 0.5, 0.5])
    edge_index, edge_weight = add_self_loops(edge_index, edge_weight)
    assert edge_index.tolist() == expected
    assert edge_weight.tolist() == [0.5, 0.5, 0.5, 1, 1]


def test_add_remaining_self_loops():
    edge_index = torch.tensor([[0, 1, 0], [1, 0, 0]])
    edge_weight = torch.tensor([0.5, 0.5, 0.5])

    edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight)
    assert edge_index.tolist() == [[0, 1, 0, 1], [1, 0, 0, 1]]
    assert edge_weight.tolist() == [0.5, 0.5, 0.5, 1]


def test_add_remaining_self_loops_without_initial_loops():
    edge_index = torch.tensor([[0, 1], [1, 0]])
    edge_weight = torch.tensor([0.5, 0.5])

    edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight)
    assert edge_index.tolist() == [[0, 1, 0, 1], [1, 0, 0, 1]]
    assert edge_weight.tolist() == [0.5, 0.5, 1, 1]
