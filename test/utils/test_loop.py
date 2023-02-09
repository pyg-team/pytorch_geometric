import torch

from torch_geometric.utils import (
    add_remaining_self_loops,
    add_self_loops,
    contains_self_loops,
    get_self_loop_attr,
    remove_self_loops,
    segregate_self_loops,
)


def test_contains_self_loops():
    edge_index = torch.tensor([[0, 1, 0], [1, 0, 0]])
    assert contains_self_loops(edge_index)

    edge_index = torch.tensor([[0, 1, 1], [1, 0, 2]])
    assert not contains_self_loops(edge_index)


def test_remove_self_loops():
    edge_index = torch.tensor([[0, 1, 0], [1, 0, 0]])
    edge_attr = [[1, 2], [3, 4], [5, 6]]
    edge_attr = torch.Tensor(edge_attr)

    expected = [[0, 1], [1, 0]]

    edge_index_out, edge_attr_out = remove_self_loops(edge_index, edge_attr)
    assert edge_index_out.tolist() == expected
    assert edge_attr_out.tolist() == [[1, 2], [3, 4]]

    edge_index_out, edge_attr_out = remove_self_loops(edge_index)
    assert edge_index_out.tolist() == expected
    assert edge_attr_out is None


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
    edge_weight = torch.tensor([0.5, 0.5, 0.5])
    edge_attr = torch.eye(3).float()

    expected = [[0, 1, 0, 0, 1], [1, 0, 0, 0, 1]]
    assert add_self_loops(edge_index)[0].tolist() == expected

    out = add_self_loops(edge_index, edge_weight)
    assert out[0].tolist() == expected
    assert out[1].tolist() == [0.5, 0.5, 0.5, 1., 1.]

    out = add_self_loops(edge_index, edge_weight, fill_value=6)
    assert out[0].tolist() == expected
    assert out[1].tolist() == [0.5, 0.5, 0.5, 6.0, 6.0]

    out = add_self_loops(edge_index, edge_weight, fill_value=torch.tensor(2.))
    assert out[0].tolist() == expected
    assert out[1].tolist() == [0.5, 0.5, 0.5, 2., 2.]

    out = add_self_loops(edge_index, edge_weight, fill_value='add')
    assert out[0].tolist() == expected
    assert out[1].tolist() == [0.5, 0.5, 0.5, 1, 0.5]

    # Tests with 'edge_attr':
    out = add_self_loops(edge_index, edge_attr)
    assert out[0].tolist() == expected
    assert out[1].tolist() == [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
                               [1., 1., 1.], [1., 1., 1.]]

    out = add_self_loops(edge_index, edge_attr,
                         fill_value=torch.tensor([0., 1., 0.]))
    assert out[0].tolist() == expected
    assert out[1].tolist() == [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
                               [0., 1., 0.], [0., 1., 0.]]

    out = add_self_loops(edge_index, edge_attr, fill_value='add')
    assert out[0].tolist() == expected
    assert out[1].tolist() == [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
                               [0., 1., 1.], [1., 0., 0.]]

    # Test empty 'edge_index' and 'edge_weight'.
    edge_index = torch.tensor([]).view(2, 0)
    edge_weight = torch.tensor([], dtype=torch.float)
    out = add_self_loops(edge_index, edge_weight, num_nodes=1)
    assert out[0].tolist() == [[0], [0]]
    assert out[1].tolist() == [1.]


def test_add_remaining_self_loops():
    edge_index = torch.tensor([[0, 1, 0], [1, 0, 0]])
    edge_weight = torch.tensor([0.5, 0.5, 0.5])
    edge_attr = torch.eye(3).float()

    expected = [[0, 1, 0, 1], [1, 0, 0, 1]]

    edge_index_out, edge_weight_out = add_remaining_self_loops(
        edge_index, edge_weight)
    assert edge_index_out.tolist() == expected
    assert edge_weight_out.tolist() == [0.5, 0.5, 0.5, 1]

    # fill_value as int
    fill_value = 5
    edge_index_out, edge_weight_out = add_remaining_self_loops(
        edge_index, edge_weight, fill_value)
    assert edge_index_out.tolist() == expected
    assert edge_weight_out.tolist() == [0.5, 0.5, 0.5, 5.0]

    # fill_value as Tensor
    fill_value = torch.Tensor([2])
    edge_index_out, edge_weight_out = add_remaining_self_loops(
        edge_index, edge_weight, fill_value)
    assert edge_index_out.tolist() == expected
    assert edge_weight_out.tolist() == [0.5, 0.5, 0.5, 2.0]

    # fill_value as str
    fill_value = "add"
    edge_index_out, edge_weight_out = add_remaining_self_loops(
        edge_index, edge_weight, fill_value)
    assert edge_index_out.tolist() == expected
    assert edge_weight_out.tolist() == [0.5, 0.5, 0.5, 0.5]

    # Test with 'edge_attr':
    fill_value = torch.Tensor([0, 1, 0])
    edge_index_out, edge_attr_out = add_remaining_self_loops(
        edge_index, edge_attr, fill_value)
    assert edge_index_out.tolist() == expected
    assert edge_attr_out.tolist() == [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
                                      [0., 1., 0.]]


def test_add_remaining_self_loops_without_initial_loops():
    edge_index = torch.tensor([[0, 1], [1, 0]])
    edge_weight = torch.tensor([0.5, 0.5])

    expected = [[0, 1, 0, 1], [1, 0, 0, 1]]

    edge_index_out, edge_weight_out = add_remaining_self_loops(
        edge_index, edge_weight)
    assert edge_index_out.tolist() == expected
    assert edge_weight_out.tolist() == [0.5, 0.5, 1, 1]

    # fill_value as int
    fill_value = 5
    edge_index_out, edge_weight_out = add_remaining_self_loops(
        edge_index, edge_weight, fill_value)
    assert edge_index_out.tolist() == expected
    assert edge_weight_out.tolist() == [0.5, 0.5, 5.0, 5.0]

    # fill_value as Tensor
    fill_value = torch.Tensor([2])
    edge_index_out, edge_weight_out = add_remaining_self_loops(
        edge_index, edge_weight, fill_value)
    assert edge_index_out.tolist() == expected
    assert edge_weight_out.tolist() == [0.5, 0.5, 2.0, 2.0]

    # fill_value as str
    fill_value = "add"
    edge_index_out, edge_weight_out = add_remaining_self_loops(
        edge_index, edge_weight, fill_value)
    assert edge_index_out.tolist() == expected
    assert edge_weight_out.tolist() == [0.5, 0.5, 0.5, 0.5]


def test_get_self_loop_attr():
    edge_index = torch.tensor([[0, 1, 0], [1, 0, 0]])
    edge_weight = torch.tensor([0.2, 0.3, 0.5])

    full_loop_weight = get_self_loop_attr(edge_index, edge_weight)
    assert full_loop_weight.tolist() == [0.5, 0.0]

    full_loop_weight = get_self_loop_attr(edge_index, edge_weight, num_nodes=4)
    assert full_loop_weight.tolist() == [0.5, 0.0, 0.0, 0.0]

    full_loop_weight = get_self_loop_attr(edge_index)
    assert full_loop_weight.tolist() == [1.0, 0.0]

    edge_attr = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 1.0]])
    full_loop_attr = get_self_loop_attr(edge_index, edge_attr)
    assert full_loop_attr.tolist() == [[0.5, 1.0], [0.0, 0.0]]
