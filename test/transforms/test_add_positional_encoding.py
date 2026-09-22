import torch

from torch_geometric.data import Data
from torch_geometric.testing import withPackage
from torch_geometric.transforms import (
    AddLaplacianEigenvectorPE,
    AddRandomWalkPE,
)


@withPackage('scipy')
def test_add_laplacian_eigenvector_pe():
    x = torch.randn(6, 4)
    edge_index = torch.tensor([[0, 1, 0, 4, 1, 4, 2, 3, 3, 5],
                               [1, 0, 4, 0, 4, 1, 3, 2, 5, 3]])
    data = Data(x=x, edge_index=edge_index)

    transform = AddLaplacianEigenvectorPE(k=3)
    assert str(transform) == 'AddLaplacianEigenvectorPE()'
    out = transform(data)
    assert out.laplacian_eigenvector_pe.size() == (6, 3)

    transform = AddLaplacianEigenvectorPE(k=3, attr_name=None)
    out = transform(data)
    assert out.x.size() == (6, 4 + 3)

    transform = AddLaplacianEigenvectorPE(k=3, attr_name='x')
    out = transform(data)
    assert out.x.size() == (6, 3)

    # Output tests:
    edge_index = torch.tensor([[0, 1, 0, 4, 1, 4, 2, 3, 3, 5, 2, 5],
                               [1, 0, 4, 0, 4, 1, 3, 2, 5, 3, 5, 2]])
    data = Data(x=x, edge_index=edge_index)

    transform1 = AddLaplacianEigenvectorPE(k=1, is_undirected=True)
    transform2 = AddLaplacianEigenvectorPE(k=1, is_undirected=False)

    # Clustering test with first non-trivial eigenvector (Fiedler vector)
    pe = transform1(data).laplacian_eigenvector_pe
    pe_cluster_1 = pe[[0, 1, 4]]
    pe_cluster_2 = pe[[2, 3, 5]]
    assert not torch.allclose(pe_cluster_1, pe_cluster_2)
    assert torch.allclose(pe_cluster_1, pe_cluster_1.mean())
    assert torch.allclose(pe_cluster_2, pe_cluster_2.mean())

    pe = transform2(data).laplacian_eigenvector_pe
    pe_cluster_1 = pe[[0, 1, 4]]
    pe_cluster_2 = pe[[2, 3, 5]]
    assert not torch.allclose(pe_cluster_1, pe_cluster_2)
    assert torch.allclose(pe_cluster_1, pe_cluster_1.mean())
    assert torch.allclose(pe_cluster_2, pe_cluster_2.mean())


@withPackage('scipy')
def test_eigenvector_permutation_invariance():
    edge_index = torch.tensor([[0, 1, 0, 4, 1, 4, 2, 3, 3, 5],
                               [1, 0, 4, 0, 4, 1, 3, 2, 5, 3]])
    data = Data(edge_index=edge_index, num_nodes=6)

    perm = torch.randperm(data.num_nodes)
    transform = AddLaplacianEigenvectorPE(
        k=2,
        is_undirected=True,
        attr_name='x',
    )
    out1 = transform(data)

    transform = AddLaplacianEigenvectorPE(
        k=2,
        is_undirected=True,
        attr_name='x',
    )
    out2 = transform(data.subgraph(perm))

    assert torch.allclose(out1.x[perm].abs(), out2.x.abs(), atol=1e-6)


def test_add_random_walk_pe():
    x = torch.randn(6, 4)
    edge_index = torch.tensor([[0, 1, 0, 4, 1, 4, 2, 3, 3, 5],
                               [1, 0, 4, 0, 4, 1, 3, 2, 5, 3]])
    data = Data(x=x, edge_index=edge_index)

    transform = AddRandomWalkPE(walk_length=3)
    assert str(transform) == 'AddRandomWalkPE()'
    out = transform(data)
    assert out.random_walk_pe.size() == (6, 3)

    transform = AddRandomWalkPE(walk_length=3, attr_name=None)
    out = transform(data)
    assert out.x.size() == (6, 4 + 3)

    transform = AddRandomWalkPE(walk_length=3, attr_name='x')
    out = transform(data)
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
    out = transform(data)

    assert out.x.tolist() == [
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0],
    ]


def test_add_random_walk_pe_edge_weight():
    # Node 0 moves to node 1 with probability 1/4 and to node 2 with 3/4, and
    # both walk straight back, so the return probabilities are exact (#10098):
    edge_index = torch.tensor([[0, 0, 1, 2], [1, 2, 0, 0]])
    edge_weight = torch.tensor([1.0, 3.0, 2.0, 2.0])
    transform = AddRandomWalkPE(walk_length=3, attr_name='pe')

    out = transform(
        Data(edge_index=edge_index, edge_weight=edge_weight, num_nodes=3))
    expected = torch.tensor([
        [0.0, 1.0, 0.0],
        [0.0, 0.25, 0.0],
        [0.0, 0.75, 0.0],
    ])
    assert torch.allclose(out.pe, expected)

    # Only the relative weights matter, including weights below one:
    out = transform(
        Data(edge_index=edge_index, edge_weight=0.1 * edge_weight,
             num_nodes=3))
    assert torch.allclose(out.pe, expected)


def test_add_random_walk_pe_duplicated_edges():
    # A duplicated edge counts once per copy, on the dense code path (at most
    # 2,000 nodes) as on the sparse one, so padding the graph with isolated
    # nodes past that threshold must not change the encoding:
    edge_index = torch.tensor([[0, 0, 0, 1, 2], [1, 1, 2, 0, 0]])
    transform = AddRandomWalkPE(walk_length=4, attr_name='pe')

    dense = transform(Data(edge_index=edge_index, num_nodes=3)).pe
    sparse = transform(Data(edge_index=edge_index, num_nodes=2_001)).pe[:3]
    assert torch.allclose(dense, sparse)
    # Node 0 moves to node 1 with probability 2/3:
    assert torch.allclose(dense[1, 1], torch.tensor(2.0 / 3.0))
