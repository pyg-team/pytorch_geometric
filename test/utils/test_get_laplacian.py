from torch_geometric.utils import get_laplacian
from torch_geometric.utils import to_scipy_sparse_matrix

import torch
import numpy as np


def test_get_laplacian():
    # Create fake data
    edge_index = torch.tensor([[0, 1, 1, 2],
                               [1, 0, 2, 1]], dtype=torch.long)
    edge_weight = torch.tensor([1, 1, 2, 3], dtype=torch.float)
    num_nodes = 3
    params = {
        'edge_index': edge_index,
        'edge_weight': edge_weight,
        'num_nodes': num_nodes,
    }

    # Compute results with PyG
    lap = get_laplacian(**params)
    lap_sym = get_laplacian(**params, normalization='sym')
    lap_rw = get_laplacian(**params, normalization='rw')

    lap = to_scipy_sparse_matrix(*lap).toarray()
    lap_sym = to_scipy_sparse_matrix(*lap_sym).toarray()
    lap_rw = to_scipy_sparse_matrix(*lap_rw).toarray()

    # Compute results with numpy
    adj = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes).toarray()
    d = adj.sum(axis=1)
    deg = np.diag(d)
    deg_sym = np.diag(np.power(d, -0.5))
    deg_rw = np.diag(np.power(d, -1.0))

    # Test
    eye = np.eye(adj.shape[0])
    assert np.array_equal(lap, deg - adj)
    assert np.array_equal(lap_sym, eye - deg_sym @ adj @ deg_sym)
    assert np.array_equal(lap_rw, eye - deg_rw @ adj)
