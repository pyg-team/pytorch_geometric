import torch

from torch_geometric.utils import (
    add_remaining_self_loops,
    add_self_loops,
    contains_self_loops,
    get_self_loop_attr,
    remove_self_loops,
    segregate_self_loops,
    to_torch_coo_tensor,
)


def test_contains_self_loops():
    edge_index = torch.tensor([[0, 1, 0], [1, 0, 0]])
    assert contains_self_loops(edge_index)

    edge_index = torch.tensor([[0, 1, 1], [1, 0, 2]])
    assert not contains_self_loops(edge_index)


def test_remove_self_loops():
    edge_index = torch.tensor([[0, 1, 0], [1, 0, 0]])
    edge_attr = torch.tensor([[1, 2], [3, 4], [5, 6]])

    expected = [[0, 1], [1, 0]]

    out = remove_self_loops(edge_index)
    assert out[0].tolist() == expected
    assert out[1] is None

    out = remove_self_loops(edge_index, edge_attr)
    assert out[0].tolist() == expected
    assert out[1].tolist() == [[1, 2], [3, 4]]

    adj = to_torch_coo_tensor(edge_index)
    adj, _ = remove_self_loops(adj)
    assert torch.diag(adj.to_dense()).tolist() == [0, 0]


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
    edge_attr = torch.eye(3)
    adj = to_torch_coo_tensor(edge_index, edge_weight)

    expected = [[0, 1, 0, 0, 1], [1, 0, 0, 0, 1]]
    assert add_self_loops(edge_index)[0].tolist() == expected

    out = add_self_loops(edge_index, edge_weight)
    assert out[0].tolist() == expected
    assert out[1].tolist() == [0.5, 0.5, 0.5, 1., 1.]

    out = add_self_loops(adj)[0]
    assert out._indices().tolist() == [[0, 0, 1, 1], [0, 1, 0, 1]]
    assert out._values().tolist() == [1.5, 0.5, 0.5, 1.0]

    out = add_self_loops(edge_index, edge_weight, fill_value=5)
    assert out[0].tolist() == expected
    assert out[1].tolist() == [0.5, 0.5, 0.5, 5.0, 5.0]

    out = add_self_loops(adj, fill_value=5)[0]
    assert out._indices().tolist() == [[0, 0, 1, 1], [0, 1, 0, 1]]
    assert out._values().tolist() == [5.5, 0.5, 0.5, 5.0]

    out = add_self_loops(edge_index, edge_weight, fill_value=torch.tensor(2.))
    assert out[0].tolist() == expected
    assert out[1].tolist() == [0.5, 0.5, 0.5, 2., 2.]

    out = add_self_loops(adj, fill_value=torch.tensor(2.))[0]
    assert out._indices().tolist() == [[0, 0, 1, 1], [0, 1, 0, 1]]
    assert out._values().tolist() == [2.5, 0.5, 0.5, 2.0]

    out = add_self_loops(edge_index, edge_weight, fill_value='add')
    assert out[0].tolist() == expected
    assert out[1].tolist() == [0.5, 0.5, 0.5, 1, 0.5]

    # Tests with `edge_attr`:
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

    # Test empty `edge_index` and `edge_weight`:
    edge_index = torch.empty(2, 0, dtype=torch.long)
    edge_weight = torch.empty(0)
    out = add_self_loops(edge_index, edge_weight, num_nodes=1)
    assert out[0].tolist() == [[0], [0]]
    assert out[1].tolist() == [1.]


def test_add_remaining_self_loops():
    edge_index = torch.tensor([[0, 1, 0], [1, 0, 0]])
    edge_weight = torch.tensor([0.5, 0.5, 0.5])
    edge_attr = torch.eye(3)

    expected = [[0, 1, 0, 1], [1, 0, 0, 1]]

    out = add_remaining_self_loops(edge_index, edge_weight)
    assert out[0].tolist() == expected
    assert out[1].tolist() == [0.5, 0.5, 0.5, 1]

    out = add_remaining_self_loops(edge_index, edge_weight, fill_value=5)
    assert out[0].tolist() == expected
    assert out[1].tolist() == [0.5, 0.5, 0.5, 5.0]

    out = add_remaining_self_loops(edge_index, edge_weight,
                                   fill_value=torch.tensor(2.))
    assert out[0].tolist() == expected
    assert out[1].tolist() == [0.5, 0.5, 0.5, 2.0]

    out = add_remaining_self_loops(edge_index, edge_weight, fill_value='add')
    assert out[0].tolist() == expected
    assert out[1].tolist() == [0.5, 0.5, 0.5, 0.5]

    # Test with `edge_attr`:
    out = add_remaining_self_loops(edge_index, edge_attr,
                                   fill_value=torch.tensor([0., 1., 0.]))
    assert out[0].tolist() == expected
    assert out[1].tolist() == [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
                               [0., 1., 0.]]


def test_add_remaining_self_loops_without_initial_loops():
    edge_index = torch.tensor([[0, 1], [1, 0]])
    edge_weight = torch.tensor([0.5, 0.5])

    expected = [[0, 1, 0, 1], [1, 0, 0, 1]]

    out = add_remaining_self_loops(edge_index, edge_weight)
    assert out[0].tolist() == expected
    assert out[1].tolist() == [0.5, 0.5, 1, 1]

    out = add_remaining_self_loops(edge_index, edge_weight, fill_value=5)
    assert out[0].tolist() == expected
    assert out[1].tolist() == [0.5, 0.5, 5.0, 5.0]

    out = add_remaining_self_loops(edge_index, edge_weight,
                                   fill_value=torch.tensor(2.0))
    assert out[0].tolist() == expected
    assert out[1].tolist() == [0.5, 0.5, 2.0, 2.0]

    # Test string `fill_value`:
    out = add_remaining_self_loops(edge_index, edge_weight, fill_value='add')
    assert out[0].tolist() == expected
    assert out[1].tolist() == [0.5, 0.5, 0.5, 0.5]


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
