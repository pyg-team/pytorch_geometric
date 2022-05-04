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
    assert str(transform_attr) == 'AddLaplacianEigenvectorPE()'

    data = Data(x=x, edge_index=edge_index)
    out = transform_attr(data)
    assert hasattr(out, 'laplacian_eigenvector_pe')
    assert out.laplacian_eigenvector_pe.size() == (N, num_pe)

    transform_cat = AddLaplacianEigenvectorPE(num_pe, 'cat')

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

    transform_cat = AddRandomWalkPE(num_pe, 'cat')
    assert str(transform_cat) == 'AddRandomWalkPE()'

    data = Data(x=x, edge_index=edge_index)
    out = transform_cat(data)
    assert out.x.size() == (N, num_pe + F)

    data = Data(edge_index=edge_index)
    out = transform_cat(data)
    assert out.x.size() == (N, num_pe)

    transform_attr = AddRandomWalkPE(num_pe, 'attr')
    data = Data(x=x, edge_index=edge_index)
    out = transform_attr(data)
    assert hasattr(out, 'random_walk_pe')
    assert out.random_walk_pe.size() == (N, num_pe)
    assert out.random_walk_pe.tolist() == [[0.0, 0.5, 0.25], [0.0, 0.5, 0.25],
                                           [0.0, 0.5, 0.0], [0.0, 1.0, 0.0],
                                           [0.0, 0.5, 0.25], [0.0, 0.5, 0.0]]

    edge_index = torch.tensor([[0, 1, 2], [0, 1, 2]])
    data = Data(edge_index=edge_index, num_nodes=4)
    out = transform_attr(data)
    assert out.random_walk_pe.tolist() == [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0],
                                           [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]
