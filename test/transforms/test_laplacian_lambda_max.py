import pytest
import torch

from torch_geometric.data import Data
from torch_geometric.testing import withPackage
from torch_geometric.transforms import LaplacianLambdaMax


@withPackage('scipy')
@pytest.mark.parametrize('normalization', [None, 'sym', 'rw'])
@pytest.mark.parametrize('is_undirected', [False, True])
@pytest.mark.parametrize('num_nodes,connected', [(1, False), (2, False),
                                                 (2, True)])
def test_laplacian_lambda_max_small_graph(normalization, is_undirected,
                                          num_nodes, connected):
    if connected:
        edge_index = torch.tensor([[0, 1], [1, 0]])
        edge_attr = torch.tensor([3., 3.])
        expected = 6. if normalization is None else 2.
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty(0)
        expected = 0. if normalization is None else 1.

    data = Data(edge_index=edge_index, edge_attr=edge_attr,
                num_nodes=num_nodes)
    transform = LaplacianLambdaMax(normalization, is_undirected)
    assert transform(data).lambda_max == pytest.approx(expected)


@withPackage('scipy')
def test_laplacian_lambda_max():
    out = str(LaplacianLambdaMax())
    assert out == 'LaplacianLambdaMax(normalization=None)'

    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    edge_attr = torch.tensor([1, 1, 2, 2], dtype=torch.float)

    data = Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=3)
    out = LaplacianLambdaMax(normalization=None, is_undirected=True)(data)
    assert len(out) == 4
    assert torch.allclose(torch.tensor(out.lambda_max), torch.tensor(4.732049))

    data = Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=3)
    out = LaplacianLambdaMax(normalization='sym', is_undirected=True)(data)
    assert len(out) == 4
    assert torch.allclose(torch.tensor(out.lambda_max), torch.tensor(2.0))

    data = Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=3)
    out = LaplacianLambdaMax(normalization='rw', is_undirected=True)(data)
    assert len(out) == 4
    assert torch.allclose(torch.tensor(out.lambda_max), torch.tensor(2.0))

    data = Data(edge_index=edge_index, edge_attr=torch.randn(4, 2),
                num_nodes=3)
    out = LaplacianLambdaMax(normalization=None)(data)
    assert len(out) == 4
    assert torch.allclose(torch.tensor(out.lambda_max), torch.tensor(3.0))
