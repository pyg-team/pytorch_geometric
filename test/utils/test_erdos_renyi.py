import torch
from torch_geometric.utils import erdos_renyi_graph


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
