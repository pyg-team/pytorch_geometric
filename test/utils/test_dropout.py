import pytest
import torch

from torch_geometric.testing import withPackage
from torch_geometric.utils import (
    dropout_adj,
    dropout_edge,
    dropout_node,
    dropout_path,
)


def test_dropout_adj():
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3],
        [1, 0, 2, 1, 3, 2],
    ])
    edge_attr = torch.Tensor([1, 2, 3, 4, 5, 6])

    with pytest.warns(UserWarning, match="'dropout_adj' is deprecated"):
        out = dropout_adj(edge_index, edge_attr, training=False)
    assert edge_index.tolist() == out[0].tolist()
    assert edge_attr.tolist() == out[1].tolist()

    torch.manual_seed(5)
    with pytest.warns(UserWarning, match="'dropout_adj' is deprecated"):
        out = dropout_adj(edge_index, edge_attr)
    assert out[0].tolist() == [[0, 1, 2, 2], [1, 2, 1, 3]]
    assert out[1].tolist() == [1, 3, 4, 5]

    torch.manual_seed(6)
    with pytest.warns(UserWarning, match="'dropout_adj' is deprecated"):
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


@withPackage('torch_cluster')
def test_dropout_path():
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])

    out = dropout_path(edge_index, training=False)
    assert edge_index.tolist() == out[0].tolist()
    assert out[1].tolist() == [True, True, True, True, True, True]

    torch.manual_seed(4)
    out = dropout_path(edge_index, p=0.2)
    assert out[0].tolist() == [[0, 1], [1, 0]]
    assert out[1].tolist() == [True, True, False, False, False, False]
    assert edge_index[:, out[1]].tolist() == out[0].tolist()

    # test with unsorted edges
    torch.manual_seed(5)
    edge_index = torch.tensor([[3, 5, 2, 2, 2, 1], [1, 0, 0, 1, 3, 2]])
    out = dropout_path(edge_index, p=0.2)
    assert out[0].tolist() == [[3, 2, 2, 1], [1, 1, 3, 2]]
    assert out[1].tolist() == [True, False, False, True, True, True]
    assert edge_index[:, out[1]].tolist() == out[0].tolist()

    # test with isolated nodes
    torch.manual_seed(7)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 2, 4]])
    out = dropout_path(edge_index, p=0.2)
    assert out[0].tolist() == [[2, 3], [2, 4]]
    assert out[1].tolist() == [False, False, True, True]
    assert edge_index[:, out[1]].tolist() == out[0].tolist()
