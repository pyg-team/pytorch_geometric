import torch

from torch_geometric import EdgeIndex
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

    expected = torch.tensor([[0, 1], [1, 0]])

    out = remove_self_loops(edge_index)
    assert out[0].equal(expected)
    assert out[1] is None

    out = remove_self_loops(edge_index, edge_attr)
    assert out[0].equal(expected)
    assert out[1].equal(torch.tensor([[1, 2], [3, 4]]))

    adj = to_torch_coo_tensor(edge_index)
    adj, _ = remove_self_loops(adj)
    assert torch.diag(adj.to_dense()).tolist() == [0, 0]

    edge_index = EdgeIndex(
        edge_index,
        sparse_size=(2, 2),
        sort_order='row',
        is_undirected=True,
    )
    out = remove_self_loops(edge_index)
    assert out[0].equal(expected)
    assert out[0].sparse_size() == (2, 2)
    assert out[0].sort_order == 'row'
    assert out[0].is_undirected
    assert out[1] is None

    out = remove_self_loops(edge_index, edge_attr)
    assert out[0].equal(expected)
    assert out[0].sparse_size() == (2, 2)
    assert out[0].sort_order == 'row'
    assert out[0].is_undirected
    assert out[1].equal(torch.tensor([[1, 2], [3, 4]]))


def test_segregate_self_loops():
    edge_index = torch.tensor([[0, 0, 1], [0, 1, 0]])

    out = segregate_self_loops(edge_index)
    assert out[0].equal(torch.tensor([[0, 1], [1, 0]]))
    assert out[1] is None
    assert out[2].equal(torch.tensor([[0], [0]]))
    assert out[3] is None

    edge_attr = torch.tensor([1, 2, 3])
    out = segregate_self_loops(edge_index, edge_attr)
    assert out[0].equal(torch.tensor([[0, 1], [1, 0]]))
    assert out[1].equal(torch.tensor([2, 3]))
    assert out[2].equal(torch.tensor([[0], [0]]))
    assert out[3].equal(torch.tensor([1]))

    edge_index = EdgeIndex(
        edge_index,
        sparse_size=(2, 2),
        sort_order='row',
        is_undirected=True,
    )
    out = segregate_self_loops(edge_index)
    assert out[0].equal(torch.tensor([[0, 1], [1, 0]]))
    assert out[0].sparse_size() == (2, 2)
    assert out[0].sort_order == 'row'
    assert out[0].is_undirected
    assert out[1] is None
    assert out[2].equal(torch.tensor([[0], [0]]))
    assert out[2].sparse_size() == (2, 2)
    assert out[2].sort_order == 'row'
    assert out[2].is_undirected
    assert out[3] is None

    out = segregate_self_loops(edge_index, edge_attr)
    assert out[0].equal(torch.tensor([[0, 1], [1, 0]]))
    assert out[0].sparse_size() == (2, 2)
    assert out[0].sort_order == 'row'
    assert out[0].is_undirected
    assert out[1].equal(torch.tensor([2, 3]))
    assert out[2].equal(torch.tensor([[0], [0]]))
    assert out[2].sparse_size() == (2, 2)
    assert out[2].sort_order == 'row'
    assert out[2].is_undirected
    assert out[3].equal(torch.tensor([1]))


def test_add_self_loops():
    edge_index = torch.tensor([[0, 1, 0], [1, 0, 0]])
    edge_weight = torch.tensor([0.5, 0.5, 0.5])
    edge_attr = torch.eye(3)
    adj = to_torch_coo_tensor(edge_index, edge_weight)

    expected = torch.tensor([[0, 1, 0, 0, 1], [1, 0, 0, 0, 1]])
    assert add_self_loops(edge_index)[0].equal(expected)

    out = add_self_loops(edge_index, edge_weight)
    assert out[0].equal(expected)
    assert out[1].equal(torch.tensor([0.5, 0.5, 0.5, 1., 1.]))

    out = add_self_loops(adj)[0]
    assert out._indices().equal(torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]]))
    assert out._values().equal(torch.tensor([1.5, 0.5, 0.5, 1.0]))

    out = add_self_loops(edge_index, edge_weight, fill_value=5)
    assert out[0].equal(expected)
    assert out[1].equal(torch.tensor([0.5, 0.5, 0.5, 5.0, 5.0]))

    out = add_self_loops(adj, fill_value=5)[0]
    assert out._indices().equal(torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]]))
    assert out._values().equal(torch.tensor([5.5, 0.5, 0.5, 5.0]))

    out = add_self_loops(edge_index, edge_weight, fill_value=torch.tensor(2.))
    assert out[0].equal(expected)
    assert out[1].equal(torch.tensor([0.5, 0.5, 0.5, 2., 2.]))

    out = add_self_loops(adj, fill_value=torch.tensor(2.))[0]
    assert out._indices().equal(torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]]))
    assert out._values().equal(torch.tensor([2.5, 0.5, 0.5, 2.0]))

    out = add_self_loops(edge_index, edge_weight, fill_value='add')
    assert out[0].equal(expected)
    assert out[1].equal(torch.tensor([0.5, 0.5, 0.5, 1, 0.5]))

    # Tests with `edge_attr`:
    out = add_self_loops(edge_index, edge_attr)
    assert out[0].equal(expected)
    assert out[1].equal(
        torch.tensor([
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
        ]))

    out = add_self_loops(edge_index, edge_attr,
                         fill_value=torch.tensor([0., 1., 0.]))
    assert out[0].equal(expected)
    assert out[1].equal(
        torch.tensor([
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
            [0., 1., 0.],
            [0., 1., 0.],
        ]))

    out = add_self_loops(edge_index, edge_attr, fill_value='add')
    assert out[0].equal(expected)
    assert out[1].equal(
        torch.tensor([
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
            [0., 1., 1.],
            [1., 0., 0.],
        ]))

    edge_index = EdgeIndex(
        edge_index,
        sparse_size=(2, 2),
        sort_order='row',
        is_undirected=True,
    )
    out, _ = add_self_loops(edge_index)
    assert out.equal(expected)
    assert out.sparse_size() == (2, 2)
    assert out.sort_order is None
    assert out.is_undirected

    # Test empty `edge_index` and `edge_weight`:
    edge_index = torch.empty(2, 0, dtype=torch.long)
    edge_weight = torch.empty(0)
    out = add_self_loops(edge_index, edge_weight, num_nodes=1)
    assert out[0].equal(torch.tensor([[0], [0]]))
    assert out[1].equal(torch.tensor([1.]))


def test_add_self_loops_bipartite():
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    adj = to_torch_coo_tensor(edge_index, size=(4, 2))

    edge_index, _ = add_self_loops(edge_index, num_nodes=(4, 2))
    assert edge_index.equal(
        torch.tensor([
            [0, 1, 2, 3, 0, 1],
            [0, 0, 1, 1, 0, 1],
        ]))

    adj, _ = add_self_loops(adj)
    assert adj._indices().equal(
        torch.tensor([
            [0, 1, 1, 2, 3],
            [0, 0, 1, 1, 1],
        ]))


def test_add_remaining_self_loops():
    edge_index = torch.tensor([[0, 1, 0], [1, 0, 0]])
    edge_weight = torch.tensor([0.5, 0.5, 0.5])
    edge_attr = torch.eye(3)

    expected = torch.tensor([[0, 1, 0, 1], [1, 0, 0, 1]])

    out = add_remaining_self_loops(edge_index, edge_weight)
    assert out[0].equal(expected)
    assert out[1].equal(torch.tensor([0.5, 0.5, 0.5, 1]))

    out = add_remaining_self_loops(edge_index, edge_weight, fill_value=5)
    assert out[0].equal(expected)
    assert out[1].equal(torch.tensor([0.5, 0.5, 0.5, 5.0]))

    out = add_remaining_self_loops(edge_index, edge_weight,
                                   fill_value=torch.tensor(2.))
    assert out[0].equal(expected)
    assert out[1].equal(torch.tensor([0.5, 0.5, 0.5, 2.0]))

    out = add_remaining_self_loops(edge_index, edge_weight, fill_value='add')
    assert out[0].equal(expected)
    assert out[1].equal(torch.tensor([0.5, 0.5, 0.5, 0.5]))

    # Test with `edge_attr`:
    out = add_remaining_self_loops(edge_index, edge_attr,
                                   fill_value=torch.tensor([0., 1., 0.]))
    assert out[0].equal(expected)
    assert out[1].equal(
        torch.tensor([
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
            [0., 1., 0.],
        ]))

    edge_index = EdgeIndex(
        edge_index,
        sparse_size=(2, 2),
        sort_order='row',
        is_undirected=True,
    )
    out, _ = add_remaining_self_loops(edge_index)
    assert out.equal(expected)
    assert out.sparse_size() == (2, 2)
    assert out.sort_order is None
    assert out.is_undirected


def test_add_remaining_self_loops_without_initial_loops():
    edge_index = torch.tensor([[0, 1], [1, 0]])
    edge_weight = torch.tensor([0.5, 0.5])

    expected = torch.tensor([[0, 1, 0, 1], [1, 0, 0, 1]])

    out = add_remaining_self_loops(edge_index, edge_weight)
    assert out[0].equal(expected)
    assert out[1].equal(torch.tensor([0.5, 0.5, 1, 1]))

    out = add_remaining_self_loops(edge_index, edge_weight, fill_value=5)
    assert out[0].equal(expected)
    assert out[1].equal(torch.tensor([0.5, 0.5, 5.0, 5.0]))

    out = add_remaining_self_loops(edge_index, edge_weight,
                                   fill_value=torch.tensor(2.0))
    assert out[0].equal(expected)
    assert out[1].equal(torch.tensor([0.5, 0.5, 2.0, 2.0]))

    # Test string `fill_value`:
    out = add_remaining_self_loops(edge_index, edge_weight, fill_value='add')
    assert out[0].equal(expected)
    assert out[1].equal(torch.tensor([0.5, 0.5, 0.5, 0.5]))


def test_get_self_loop_attr():
    edge_index = torch.tensor([[0, 1, 0], [1, 0, 0]])
    edge_weight = torch.tensor([0.2, 0.3, 0.5])

    full_loop_weight = get_self_loop_attr(edge_index, edge_weight)
    assert full_loop_weight.equal(torch.tensor([0.5, 0.0]))

    full_loop_weight = get_self_loop_attr(edge_index, edge_weight, num_nodes=4)
    assert full_loop_weight.equal(torch.tensor([0.5, 0.0, 0.0, 0.0]))

    full_loop_weight = get_self_loop_attr(edge_index)
    assert full_loop_weight.equal(torch.tensor([1.0, 0.0]))

    edge_attr = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 1.0]])
    full_loop_attr = get_self_loop_attr(edge_index, edge_attr)
    assert full_loop_attr.equal(torch.tensor([[0.5, 1.0], [0.0, 0.0]]))
