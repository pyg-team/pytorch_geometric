import torch
from torch_geometric.utils import (erdos_renyi_graph,
                                   stochastic_blockmodel_graph)


def test_erdos_renyi_graph():
    torch.manual_seed(1234)
    edge_index = erdos_renyi_graph(5, 0.2, directed=False)
    assert edge_index.tolist() == [
        [0, 1, 1, 1, 2, 4],
        [1, 0, 2, 4, 1, 1],
    ]

    edge_index = erdos_renyi_graph(5, 0.5, directed=True)
    assert edge_index.tolist() == [
        [1, 1, 2, 2, 3, 4, 4, 4],
        [0, 3, 0, 4, 0, 0, 1, 3],
    ]


def test_stochastic_blockmodel_graph():
    torch.manual_seed(12345)

    block_sizes = [2, 2, 4]
    edge_probs = [
        [0.25, 0.05, 0.02],
        [0.05, 0.35, 0.07],
        [0.02, 0.07, 0.40],
    ]

    edge_index = stochastic_blockmodel_graph(
        block_sizes, edge_probs, directed=False)
    assert edge_index.tolist() == [
        [2, 3, 4, 4, 5, 5, 6, 7, 7, 7],
        [3, 2, 5, 7, 4, 7, 7, 4, 5, 6],
    ]

    edge_index = stochastic_blockmodel_graph(
        block_sizes, edge_probs, directed=True)
    assert edge_index.tolist() == [
        [0, 1, 3, 5, 6, 6, 7, 7],
        [3, 3, 2, 4, 4, 7, 5, 6],
    ]
