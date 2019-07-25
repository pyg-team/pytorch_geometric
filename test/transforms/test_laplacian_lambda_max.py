import torch
import numpy as np

from torch_geometric.transforms import LaplacianLambdaMax
from torch_geometric.data import Data


def test_laplacian_lambda_max():
    # Create fake data
    x = torch.tensor([1, 2, 3], dtype=torch.float).t()

    # Test 1 - no edge_index
    trans = LaplacianLambdaMax()
    data = Data()
    data = trans(data)
    assert repr(trans) == 'LaplacianLambdaMax(lambda_max=None)'
    assert not hasattr(data, 'lambda_max')

    # Test 2 - pre-set lambda_max
    data = Data()
    lambda_max = 3.0
    trans = LaplacianLambdaMax(lambda_max=lambda_max)
    data = trans(data)
    assert repr(trans) == 'LaplacianLambdaMax(lambda_max=3.0)'
    assert hasattr(data, 'lambda_max') and data.lambda_max == lambda_max

    # Test 3 - non-symmetric adjacency matrix
    edge_index = torch.tensor([[0, 1, 1, 2],
                               [1, 0, 2, 1]], dtype=torch.long)
    edge_weight = torch.tensor([1, 1, 2, 3], dtype=torch.float)
    data = Data(edge_index=edge_index, edge_weight=edge_weight, x=x)

    trans = LaplacianLambdaMax()
    data = trans(data)
    assert np.allclose(data.lambda_max, 2.0)

    # Test 4 - symmetric adjacency matrix
    edge_index = torch.tensor([[0, 0, 1, 1, 2],
                               [0, 1, 0, 2, 1]], dtype=torch.long)
    edge_weight = torch.tensor([1, 5, 5, 2, 2], dtype=torch.float)
    data = Data(edge_index=edge_index, edge_weight=edge_weight, x=x)

    trans = LaplacianLambdaMax()
    data = trans(data)
    assert np.allclose(data.lambda_max, 1.887)


test_laplacian_lambda_max()
