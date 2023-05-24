from itertools import product

import pytest
import torch
from torch_cluster.testing import devices, grad_dtypes, tensor

from torch_geometric.nn.pool.approx_knn import approx_knn_graph, approx_nn


def to_set(edge_index):
    return set([(i, j) for i, j in edge_index.t().tolist()])


# @withPackage('pynndescent')
@pytest.mark.parametrize('dtype,device', product(grad_dtypes, devices))
def test_approx_knn(dtype, device):
    x = tensor([
        [-1, -1],
        [-1, +1],
        [+1, +1],
        [+1, -1],
        [-1, -1],
        [-1, +1],
        [+1, +1],
        [+1, -1],
    ], dtype, device)
    y = tensor([
        [1, 0],
        [-1, 0],
    ], dtype, device)

    batch_x = tensor([0, 0, 0, 0, 1, 1, 1, 1], torch.long, device)
    batch_y = tensor([0, 1], torch.long, device)

    edge_index = approx_nn(x, y, 2)
    assert to_set(edge_index) == set([(0, 2), (0, 3), (1, 0), (1, 1)])

    edge_index = approx_nn(x, y, 2, batch_x, batch_y)
    assert to_set(edge_index) == set([(0, 2), (0, 3), (1, 4), (1, 5)])

    if x.is_cuda:
        edge_index = approx_nn(x, y, 2, batch_x, batch_y, cosine=True)
        assert to_set(edge_index) == set([(0, 2), (0, 3), (1, 4), (1, 5)])

    # Skipping a batch
    batch_x = tensor([0, 0, 0, 0, 2, 2, 2, 2], torch.long, device)
    batch_y = tensor([0, 2], torch.long, device)
    edge_index = approx_nn(x, y, 2, batch_x, batch_y)
    assert to_set(edge_index) == set([(0, 2), (0, 3), (1, 4), (1, 5)])


# @withPackage('pynndescent')
@pytest.mark.parametrize('dtype,device', product(grad_dtypes, devices))
def test_approx_knn_graph(dtype, device):
    x = tensor([
        [-1, -1],
        [-1, +1],
        [+1, +1],
        [+1, -1],
    ], dtype, device)

    edge_index = approx_knn_graph(x, k=2, flow='target_to_source')
    assert to_set(edge_index) == set([(0, 1), (0, 3), (1, 0), (1, 2), (2, 1),
                                      (2, 3), (3, 0), (3, 2)])

    edge_index = approx_knn_graph(x, k=2, flow='source_to_target')
    assert to_set(edge_index) == set([(1, 0), (3, 0), (0, 1), (2, 1), (1, 2),
                                      (3, 2), (0, 3), (2, 3)])
