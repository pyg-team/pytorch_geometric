import torch

from torch_geometric.utils import (
    add_random_edge,
    is_undirected,
    mask_feature,
    shuffle_node,
)


def test_shuffle_node():
    x = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.float)

    out = shuffle_node(x, training=False)
    assert out[0].tolist() == x.tolist()
    assert out[1].tolist() == list(range(len(x)))

    torch.manual_seed(5)
    out = shuffle_node(x)
    assert out[0].tolist() == [[3.0, 4.0, 5.0], [0.0, 1.0, 2.0]]
    assert out[1].tolist() == [1, 0]


def test_mask_feature():
    x = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=torch.float)

    out = mask_feature(x, training=False)
    assert out[0].tolist() == x.tolist()
    assert torch.all(out[1])

    torch.manual_seed(4)
    out = mask_feature(x)
    assert out[0].tolist() == [[0.0, 1.0, 0.0], [3.0, 4.0, 0.0],
                               [6.0, 7.0, 0.0]]

    assert out[1].tolist() == [[True, True, False], [True, True, False],
                               [True, True, False]]

    torch.manual_seed(5)
    out = mask_feature(x, column_wise=False)
    assert out[0].tolist() == [[0.0, 0.0, 0.0], [3.0, 0.0, 5.0],
                               [0.0, 0.0, 8.0]]
    assert out[1].tolist() == [[False, False, False], [True, False, True],
                               [False, False, True]]


def test_add_random_edge():
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    out = add_random_edge(edge_index, training=False)
    assert out[0].tolist() == edge_index.tolist()
    assert out[1].tolist() == [[], []]

    torch.manual_seed(5)
    out = add_random_edge(edge_index, p=0.5)
    assert out[0].tolist() == [[0, 1, 1, 2, 2, 3, 3, 2, 3],
                               [1, 0, 2, 1, 3, 2, 1, 2, 2]]

    assert out[1].tolist() == [[3, 2, 3], [1, 2, 2]]

    torch.manual_seed(6)
    out = add_random_edge(edge_index, p=0.5, force_undirected=True)
    assert out[0].tolist() == [[0, 1, 1, 2, 2, 3, 2, 1, 3, 0, 2, 1],
                               [1, 0, 2, 1, 3, 2, 0, 2, 1, 2, 1, 3]]
    assert out[1].tolist() == [[2, 1, 3, 0, 2, 1], [0, 2, 1, 2, 1, 3]]
    assert is_undirected(out[0])
    assert is_undirected(out[1])

    # test with bipartite graph
    torch.manual_seed(7)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5], [2, 3, 1, 4, 2, 1]])
    out = add_random_edge(edge_index, p=0.5, num_nodes=(6, 5))
    out[0].tolist() == [[0, 1, 2, 3, 4, 5, 3, 4, 1],
                        [2, 3, 1, 4, 2, 1, 1, 3, 2]]
