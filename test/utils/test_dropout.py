import torch

from torch_geometric.utils import dropout_adj, dropout_edge, dropout_node


def test_dropout_adj():
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3],
        [1, 0, 2, 1, 3, 2],
    ])
    edge_attr = torch.Tensor([1, 2, 3, 4, 5, 6])

    out = dropout_adj(edge_index, edge_attr, training=False)
    assert edge_index.tolist() == out[0].tolist()
    assert edge_attr.tolist() == out[1].tolist()

    torch.manual_seed(5)
    out = dropout_adj(edge_index, edge_attr)
    assert out[0].tolist() == [[0, 1, 2, 2], [1, 2, 1, 3]]
    assert out[1].tolist() == [1, 3, 4, 5]

    torch.manual_seed(6)
    out = dropout_adj(edge_index, edge_attr, force_undirected=True)
    assert out[0].tolist() == [[0, 1, 1, 2], [1, 2, 0, 1]]
    assert out[1].tolist() == [1, 3, 1, 3]


def test_dropout_node():
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3],
        [1, 0, 2, 1, 3, 2],
    ])

    out = dropout_node(edge_index, training=False)
    assert edge_index.tolist() == out[0].tolist()
    assert out[1].tolist() == [True, True, True, True, True, True]
    assert out[2].tolist() == [True, True, True, True]

    torch.manual_seed(5)
    out = dropout_node(edge_index)
    assert out[0].tolist() == [[2, 3], [3, 2]]
    assert out[1].tolist() == [False, False, False, False, True, True]
    assert out[2].tolist() == [True, False, True, True]


def test_dropout_edge():
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])

    out = dropout_edge(edge_index, training=False)
    assert edge_index.tolist() == out[0].tolist()
    assert out[1].tolist() == [True, True, True, True, True, True]

    torch.manual_seed(5)
    out = dropout_edge(edge_index)
    assert out[0].tolist() == [[0, 1, 2, 2], [1, 2, 1, 3]]
    assert out[1].tolist() == [True, False, True, True, True, False]

    torch.manual_seed(6)
    out = dropout_edge(edge_index, force_undirected=True)
    assert out[0].tolist() == [[0, 1, 1, 2], [1, 2, 0, 1]]
    assert out[1].tolist() == [0, 2, 0, 2]
