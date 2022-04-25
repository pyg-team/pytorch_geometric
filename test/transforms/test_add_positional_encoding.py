import torch

from torch_geometric.data import Data
from torch_geometric.transforms import (
    AddLaplacianEigenvectorPE,
    AddRandomWalkPE,
)


def test_add_laplacian_eigenvector_pe():
    x = torch.tensor([0., 1., 2., 3., 4., 5.]).view(-1, 1)
    edge_index = torch.tensor([[0, 1, 0, 4, 1, 4, 2, 3, 3, 5],
                               [1, 0, 4, 0, 4, 1, 3, 2, 5, 3]])
    N, F = x.size()
    num_pe = 3

    transform_attr = AddLaplacianEigenvectorPE(num_pe, 'attr')
    assert transform_attr.__repr__() == 'AddLaplacianEigenvectorPE()'

    data = Data(x=x, edge_index=edge_index)
    out = transform_attr(data)
    assert hasattr(out, 'laplacian_eigenvector_pe')
    assert getattr(out, 'laplacian_eigenvector_pe').size() == (N, num_pe)

    transform_cat = AddLaplacianEigenvectorPE(num_pe, 'cat')
    assert transform_cat.__repr__() == 'AddLaplacianEigenvectorPE()'

    data = Data(x=x, edge_index=edge_index)
    out = transform_cat(data)
    assert out.x.size() == (N, num_pe + F)

    data = Data(edge_index=edge_index)
    out = transform_cat(data)
    assert out.x.size() == (N, num_pe)


def test_add_random_walk_pe():
    x = torch.tensor([0., 1., 2., 3., 4., 5.]).view(-1, 1)
    edge_index = torch.tensor([[0, 1, 0, 4, 1, 4, 2, 3, 3, 5],
                               [1, 0, 4, 0, 4, 1, 3, 2, 5, 3]])
    N, F = x.size()
    num_pe = 3

    transform_attr = AddRandomWalkPE(num_pe, 'attr')
    assert transform_attr.__repr__() == 'AddRandomWalkPE()'

    data = Data(x=x, edge_index=edge_index)
    out = transform_attr(data)
    assert hasattr(out, 'random_walk_pe')
    assert getattr(out, 'random_walk_pe').size() == (N, num_pe)

    transform_cat = AddRandomWalkPE(num_pe, 'cat')
    assert transform_cat.__repr__() == 'AddRandomWalkPE()'

    data = Data(x=x, edge_index=edge_index)
    out = transform_cat(data)
    assert out.x.size() == (N, num_pe + F)

    data = Data(edge_index=edge_index)
    out = transform_cat(data)
    assert out.x.size() == (N, num_pe)
