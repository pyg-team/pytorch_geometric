import warnings

import torch

from torch_geometric.nn import approx_knn, approx_knn_graph
from torch_geometric.testing import onlyFullTest, withPackage


def to_set(edge_index):
    return set([(i, j) for i, j in edge_index.t().tolist()])


@onlyFullTest  # JIT compile makes this test too slow :(
@withPackage('pynndescent')
def test_approx_knn():
    warnings.filterwarnings('ignore', '.*find n_neighbors.*')

    x = torch.tensor([
        [-1.0, -1.0],
        [-1.0, +1.0],
        [+1.0, +1.0],
        [+1.0, -1.0],
        [-1.0, -1.0],
        [-1.0, +1.0],
        [+1.0, +1.0],
        [+1.0, -1.0],
    ])
    y = torch.tensor([
        [+1.0, 0.0],
        [-1.0, 0.0],
    ])

    batch_x = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    batch_y = torch.tensor([0, 1])

    edge_index = approx_knn(x, y, 2)
    assert to_set(edge_index) == set([(0, 2), (0, 3), (1, 0), (1, 1)])

    edge_index = approx_knn(x, y, 2, batch_x, batch_y)
    assert to_set(edge_index) == set([(0, 2), (0, 3), (1, 4), (1, 5)])


@onlyFullTest  # JIT compile makes this test too slow :(
@withPackage('pynndescent')
def test_approx_knn_graph():
    warnings.filterwarnings('ignore', '.*find n_neighbors.*')

    x = torch.tensor([
        [-1.0, -1.0],
        [-1.0, +1.0],
        [+1.0, +1.0],
        [+1.0, -1.0],
    ])

    edge_index = approx_knn_graph(x, k=2, flow='target_to_source')
    assert to_set(edge_index) == set([(0, 1), (0, 3), (1, 0), (1, 2), (2, 1),
                                      (2, 3), (3, 0), (3, 2)])

    edge_index = approx_knn_graph(x, k=2, flow='source_to_target')
    assert to_set(edge_index) == set([(1, 0), (3, 0), (0, 1), (2, 1), (1, 2),
                                      (3, 2), (0, 3), (2, 3)])
