import torch
from torch_sparse import SparseTensor

from torch_geometric.testing import is_full_test
from torch_geometric.utils import (
    dense_to_sparse,
    is_sparse,
    is_torch_sparse_tensor,
    to_torch_coo_tensor,
)


def test_dense_to_sparse():
    adj = torch.Tensor([
        [3, 1],
        [2, 0],
    ])
    edge_index, edge_attr = dense_to_sparse(adj)
    assert edge_index.tolist() == [[0, 0, 1], [0, 1, 0]]
    assert edge_attr.tolist() == [3, 1, 2]

    if is_full_test():
        jit = torch.jit.script(dense_to_sparse)
        edge_index, edge_attr = jit(adj)
        assert edge_index.tolist() == [[0, 0, 1], [0, 1, 0]]
        assert edge_attr.tolist() == [3, 1, 2]

    adj = torch.Tensor([[
        [3, 1],
        [2, 0],
    ], [
        [0, 1],
        [0, 2],
    ]])
    edge_index, edge_attr = dense_to_sparse(adj)
    assert edge_index.tolist() == [[0, 0, 1, 2, 3], [0, 1, 0, 3, 3]]
    assert edge_attr.tolist() == [3, 1, 2, 1, 2]

    if is_full_test():
        jit = torch.jit.script(dense_to_sparse)
        edge_index, edge_attr = jit(adj)
        assert edge_index.tolist() == [[0, 0, 1, 2, 3], [0, 1, 0, 3, 3]]
        assert edge_attr.tolist() == [3, 1, 2, 1, 2]


def test_dense_to_sparse_bipartite():
    edge_index, edge_attr = dense_to_sparse(torch.rand(2, 10, 5))
    assert edge_index[0].max() == 19
    assert edge_index[1].max() == 9


def test_is_torch_sparse_tensor():
    x = torch.randn(5, 5)

    assert not is_torch_sparse_tensor(x)
    assert not is_torch_sparse_tensor(SparseTensor.from_dense(x))
    assert is_torch_sparse_tensor(x.to_sparse())


def test_is_sparse():
    x = torch.randn(5, 5)

    assert not is_sparse(x)
    assert is_sparse(SparseTensor.from_dense(x))
    assert is_sparse(x.to_sparse())


def test_to_torch_coo_tensor():
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3],
        [1, 0, 2, 1, 3, 2],
    ])
    edge_attr = torch.randn(edge_index.size(1), 8)

    adj = to_torch_coo_tensor(edge_index)
    assert adj.size() == (4, 4)
    assert adj.layout == torch.sparse_coo
    assert torch.allclose(adj.indices(), edge_index)

    adj = to_torch_coo_tensor(edge_index, size=6)
    assert adj.size() == (6, 6)
    assert adj.layout == torch.sparse_coo
    assert torch.allclose(adj.indices(), edge_index)

    adj = to_torch_coo_tensor(edge_index, edge_attr)
    assert adj.size() == (4, 4, 8)
    assert adj.layout == torch.sparse_coo
    assert torch.allclose(adj.indices(), edge_index)
    assert torch.allclose(adj.values(), edge_attr)

    if is_full_test():
        jit = torch.jit.script(to_torch_coo_tensor)
        adj = jit(edge_index, edge_attr)
        assert adj.size() == (4, 4, 8)
        assert adj.layout == torch.sparse_coo
        assert torch.allclose(adj.indices(), edge_index)
        assert torch.allclose(adj.values(), edge_attr)
