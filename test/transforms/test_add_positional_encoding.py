import numpy as np
import torch

from torch_geometric import seed_everything
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

    transform_cat = AddLaplacianEigenvectorPE(num_pe, 'cat')
    assert str(transform_cat) == 'AddLaplacianEigenvectorPE()'

    data = Data(x=x, edge_index=edge_index)
    out = transform_cat(data)
    assert out.x.size() == (N, num_pe + F)

    data = Data(edge_index=edge_index)
    out = transform_cat(data)
    assert out.x.size() == (N, num_pe)

    transform_attr = AddLaplacianEigenvectorPE(num_pe, 'attr')
    data = Data(x=x, edge_index=edge_index)
    out = transform_attr(data)
    assert hasattr(out, 'laplacian_eigenvector_pe')
    assert out.laplacian_eigenvector_pe.size() == (N, num_pe)

    # output tests
    num_pe = 1
    edge_index = torch.tensor([[0, 1, 0, 4, 1, 4, 2, 3, 3, 5, 2, 5],
                               [1, 0, 4, 0, 4, 1, 3, 2, 5, 3, 5, 2]])

    transform_undirected = AddLaplacianEigenvectorPE(num_pe, 'attr',
                                                     is_undirected=True)
    transform_directed = AddLaplacianEigenvectorPE(num_pe, 'attr',
                                                   is_undirected=False)
    data = Data(x=x, edge_index=edge_index)

    # Clustering test with first non-trivial eigenvector (Fiedler vector)
    for _ in range(5):
        pe = transform_undirected(data).laplacian_eigenvector_pe
        pe_cluster_1 = pe[[0, 1, 4]]
        pe_cluster_2 = pe[[2, 3, 5]]
        assert torch.norm(pe_cluster_1 - pe_cluster_1.mean()) < 1e-6
        assert torch.norm(pe_cluster_2 - pe_cluster_2.mean()) < 1e-6
        assert torch.norm(pe_cluster_1.mean() - pe_cluster_2.mean()) > 1e-2

        pe = transform_directed(data).laplacian_eigenvector_pe
        pe_cluster_1 = pe[[0, 1, 4]]
        pe_cluster_2 = pe[[2, 3, 5]]
        assert torch.norm(pe_cluster_1 - pe_cluster_1.mean()) < 1e-6
        assert torch.norm(pe_cluster_2 - pe_cluster_2.mean()) < 1e-6
        assert torch.norm(pe_cluster_1.mean() - pe_cluster_2.mean()) > 1e-2

    num_pe = 2
    seed_everything(0)
    transform_deterministic = AddLaplacianEigenvectorPE(
        num_pe, 'attr', is_undirected=False, v0=np.arange(6) / 6.)
    pe = transform_deterministic(data).laplacian_eigenvector_pe
    ref = torch.Tensor([[-0.2582, -0.4564],
                        [-0.2582, -0.1826],
                        [-0.5164, -0.3651],
                        [-0.5164, -0.0913],
                        [-0.2582, 0.6390],
                        [-0.5164, 0.4564]])
    assert torch.allclose(pe, ref, atol=1e-4)

    seed_everything(0)
    transform_deterministic = AddLaplacianEigenvectorPE(
        num_pe, 'attr', is_undirected=True, v0=np.arange(6) / 6.)
    pe = transform_deterministic(data).laplacian_eigenvector_pe
    ref = torch.Tensor([[-0.4930, -0.4564],
                        [-0.4930, -0.1826],
                        [-0.3005, -0.3651],
                        [-0.3005, -0.0913],
                        [-0.4930, 0.6390],
                        [-0.3005, 0.4564]])
    assert torch.allclose(pe, ref, atol=1e-4)


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

    # output tests
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
