import copy

import torch

from torch_geometric.data import Data
from torch_geometric.transforms import (
    AddLaplacianEigenvectorPE,
    AddRandomWalkPE,
)


def test_add_laplacian_eigenvector_pe():
    x = torch.randn(6, 4)
    edge_index = torch.tensor([[0, 1, 0, 4, 1, 4, 2, 3, 3, 5],
                               [1, 0, 4, 0, 4, 1, 3, 2, 5, 3]])
    data = Data(x=x, edge_index=edge_index)

    transform = AddLaplacianEigenvectorPE(k=3)
    assert str(transform) == 'AddLaplacianEigenvectorPE()'
    out = transform(copy.copy(data))
    assert out.laplacian_eigenvector_pe.size() == (6, 3)

    transform = AddLaplacianEigenvectorPE(k=3, attr_name=None)
    out = transform(copy.copy(data))
    assert out.x.size() == (6, 4 + 3)

    transform = AddLaplacianEigenvectorPE(k=3, attr_name='x')
    out = transform(copy.copy(data))
    assert out.x.size() == (6, 3)

    # Output tests:
    edge_index = torch.tensor([[0, 1, 0, 4, 1, 4, 2, 3, 3, 5, 2, 5],
                               [1, 0, 4, 0, 4, 1, 3, 2, 5, 3, 5, 2]])
    data = Data(x=x, edge_index=edge_index)

    transform1 = AddLaplacianEigenvectorPE(k=1, is_undirected=True)
    transform2 = AddLaplacianEigenvectorPE(k=1, is_undirected=False)

    # Clustering test with first non-trivial eigenvector (Fiedler vector)
    pe = transform1(copy.copy(data)).laplacian_eigenvector_pe
    pe_cluster_1 = pe[[0, 1, 4]]
    pe_cluster_2 = pe[[2, 3, 5]]
    assert not torch.allclose(pe_cluster_1, pe_cluster_2)
    assert torch.allclose(pe_cluster_1, pe_cluster_1.mean())
    assert torch.allclose(pe_cluster_2, pe_cluster_2.mean())

    pe = transform2(copy.copy(data)).laplacian_eigenvector_pe
    pe_cluster_1 = pe[[0, 1, 4]]
    pe_cluster_2 = pe[[2, 3, 5]]
    assert not torch.allclose(pe_cluster_1, pe_cluster_2)
    assert torch.allclose(pe_cluster_1, pe_cluster_1.mean())
    assert torch.allclose(pe_cluster_2, pe_cluster_2.mean())


def test_add_random_walk_pe():
    x = torch.randn(6, 4)
    edge_index = torch.tensor([[0, 1, 0, 4, 1, 4, 2, 3, 3, 5],
                               [1, 0, 4, 0, 4, 1, 3, 2, 5, 3]])
    data = Data(x=x, edge_index=edge_index)

    transform = AddRandomWalkPE(walk_length=3)
    assert str(transform) == 'AddRandomWalkPE()'
    out = transform(copy.copy(data))
    assert out.random_walk_pe.size() == (6, 3)

    transform = AddRandomWalkPE(walk_length=3, attr_name=None)
    out = transform(copy.copy(data))
    assert out.x.size() == (6, 4 + 3)

    transform = AddRandomWalkPE(walk_length=3, attr_name='x')
    out = transform(copy.copy(data))
    assert out.x.size() == (6, 3)

    # Output tests:
    assert out.x.tolist() == [
        [0.0, 0.5, 0.25],
        [0.0, 0.5, 0.25],
        [0.0, 0.5, 0.00],
        [0.0, 1.0, 0.00],
        [0.0, 0.5, 0.25],
        [0.0, 0.5, 0.00],
    ]

    edge_index = torch.tensor([[0, 1, 2], [0, 1, 2]])
    data = Data(edge_index=edge_index, num_nodes=4)
    out = transform(copy.copy(data))

    assert out.x.tolist() == [
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0],
    ]
