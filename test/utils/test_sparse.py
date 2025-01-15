import os.path as osp

import pytest
import torch

import torch_geometric.typing
from torch_geometric.io import fs
from torch_geometric.profile import benchmark
from torch_geometric.testing import is_full_test, withCUDA, withPackage
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import (
    dense_to_sparse,
    is_sparse,
    is_torch_sparse_tensor,
    to_edge_index,
    to_torch_coo_tensor,
    to_torch_csc_tensor,
    to_torch_csr_tensor,
    to_torch_sparse_tensor,
)
from torch_geometric.utils.sparse import cat


def test_dense_to_sparse():
    adj = torch.tensor([
        [3.0, 1.0],
        [2.0, 0.0],
    ])
    edge_index, edge_attr = dense_to_sparse(adj)
    assert edge_index.tolist() == [[0, 0, 1], [0, 1, 0]]
    assert edge_attr.tolist() == [3, 1, 2]

    if is_full_test():
        jit = torch.jit.script(dense_to_sparse)
        edge_index, edge_attr = jit(adj)
        assert edge_index.tolist() == [[0, 0, 1], [0, 1, 0]]
        assert edge_attr.tolist() == [3, 1, 2]

    adj = torch.tensor([[
        [3.0, 1.0],
        [2.0, 0.0],
    ], [
        [0.0, 1.0],
        [0.0, 2.0],
    ]])
    edge_index, edge_attr = dense_to_sparse(adj)
    assert edge_index.tolist() == [[0, 0, 1, 2, 3], [0, 1, 0, 3, 3]]
    assert edge_attr.tolist() == [3, 1, 2, 1, 2]

    if is_full_test():
        jit = torch.jit.script(dense_to_sparse)
        edge_index, edge_attr = jit(adj)
        assert edge_index.tolist() == [[0, 0, 1, 2, 3], [0, 1, 0, 3, 3]]
        assert edge_attr.tolist() == [3, 1, 2, 1, 2]

    adj = torch.tensor([
        [
            [3.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        [
            [0.0, 1.0, 0.0],
            [0.0, 2.0, 3.0],
            [0.0, 5.0, 0.0],
        ],
    ])
    mask = torch.tensor([[True, True, False], [True, True, True]])

    edge_index, edge_attr = dense_to_sparse(adj, mask)

    assert edge_index.tolist() == [[0, 0, 1, 2, 3, 3, 4],
                                   [0, 1, 0, 3, 3, 4, 3]]
    assert edge_attr.tolist() == [3, 1, 2, 1, 2, 3, 5]

    if is_full_test():
        jit = torch.jit.script(dense_to_sparse)
        edge_index, edge_attr = jit(adj, mask)
        assert edge_index.tolist() == [[0, 0, 1, 2, 3, 3, 4],
                                       [0, 1, 0, 3, 3, 4, 3]]
        assert edge_attr.tolist() == [3, 1, 2, 1, 2, 3, 5]


def test_dense_to_sparse_bipartite():
    edge_index, edge_attr = dense_to_sparse(torch.rand(2, 10, 5))
    assert edge_index[0].max() == 19
    assert edge_index[1].max() == 9


def test_is_torch_sparse_tensor():
    x = torch.randn(5, 5)

    assert not is_torch_sparse_tensor(x)
    assert is_torch_sparse_tensor(x.to_sparse())

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        assert not is_torch_sparse_tensor(SparseTensor.from_dense(x))


def test_is_sparse():
    x = torch.randn(5, 5)

    assert not is_sparse(x)
    assert is_sparse(x.to_sparse())

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        assert is_sparse(SparseTensor.from_dense(x))


def test_to_torch_coo_tensor():
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3],
        [1, 0, 2, 1, 3, 2],
    ])
    edge_attr = torch.randn(edge_index.size(1), 8)

    adj = to_torch_coo_tensor(edge_index, is_coalesced=False)
    assert adj.is_coalesced()
    assert adj.size() == (4, 4)
    assert adj.layout == torch.sparse_coo
    assert torch.allclose(adj.indices(), edge_index)

    adj = to_torch_coo_tensor(edge_index, is_coalesced=True)
    assert adj.is_coalesced()
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


def test_to_torch_csr_tensor():
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3],
        [1, 0, 2, 1, 3, 2],
    ])

    adj = to_torch_csr_tensor(edge_index)
    assert adj.size() == (4, 4)
    assert adj.layout == torch.sparse_csr
    assert torch.allclose(adj.to_sparse_coo().coalesce().indices(), edge_index)

    edge_weight = torch.randn(edge_index.size(1))
    adj = to_torch_csr_tensor(edge_index, edge_weight)
    assert adj.size() == (4, 4)
    assert adj.layout == torch.sparse_csr
    coo = adj.to_sparse_coo().coalesce()
    assert torch.allclose(coo.indices(), edge_index)
    assert torch.allclose(coo.values(), edge_weight)

    if torch_geometric.typing.WITH_PT20:
        edge_attr = torch.randn(edge_index.size(1), 8)
        adj = to_torch_csr_tensor(edge_index, edge_attr)
        assert adj.size() == (4, 4, 8)
        assert adj.layout == torch.sparse_csr
        coo = adj.to_sparse_coo().coalesce()
        assert torch.allclose(coo.indices(), edge_index)
        assert torch.allclose(coo.values(), edge_attr)


def test_to_torch_csc_tensor():
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3],
        [1, 0, 2, 1, 3, 2],
    ])

    adj = to_torch_csc_tensor(edge_index)
    assert adj.size() == (4, 4)
    assert adj.layout == torch.sparse_csc
    adj_coo = adj.to_sparse_coo().coalesce()
    if torch_geometric.typing.WITH_PT20:
        assert torch.allclose(adj_coo.indices(), edge_index)
    else:
        assert torch.allclose(adj_coo.indices().flip([0]), edge_index)

    edge_weight = torch.randn(edge_index.size(1))
    adj = to_torch_csc_tensor(edge_index, edge_weight)
    assert adj.size() == (4, 4)
    assert adj.layout == torch.sparse_csc
    adj_coo = adj.to_sparse_coo().coalesce()
    if torch_geometric.typing.WITH_PT20:
        assert torch.allclose(adj_coo.indices(), edge_index)
        assert torch.allclose(adj_coo.values(), edge_weight)
    else:
        perm = adj_coo.indices()[0].argsort()
        assert torch.allclose(adj_coo.indices()[:, perm], edge_index)
        assert torch.allclose(adj_coo.values()[perm], edge_weight)

    if torch_geometric.typing.WITH_PT20:
        edge_attr = torch.randn(edge_index.size(1), 8)
        adj = to_torch_csc_tensor(edge_index, edge_attr)
        assert adj.size() == (4, 4, 8)
        assert adj.layout == torch.sparse_csc
        assert torch.allclose(adj.to_sparse_coo().coalesce().indices(),
                              edge_index)
        assert torch.allclose(adj.to_sparse_coo().coalesce().values(),
                              edge_attr)


@withPackage('torch>=2.1.0')
def test_to_torch_coo_tensor_save_load(tmp_path):
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3],
        [1, 0, 2, 1, 3, 2],
    ])
    adj = to_torch_coo_tensor(edge_index, is_coalesced=False)
    assert adj.is_coalesced()

    path = osp.join(tmp_path, 'adj.t')
    torch.save(adj, path)
    adj = fs.torch_load(path)
    assert adj.is_coalesced()


def test_to_edge_index():
    adj = torch.tensor([
        [0., 1., 0., 0.],
        [1., 0., 1., 0.],
        [0., 1., 0., 1.],
        [0., 0., 1., 0.],
    ]).to_sparse()

    edge_index, edge_attr = to_edge_index(adj)
    assert edge_index.tolist() == [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]
    assert edge_attr.tolist() == [1., 1., 1., 1., 1., 1.]

    if is_full_test():
        jit = torch.jit.script(to_edge_index)
        edge_index, edge_attr = jit(adj)
        assert edge_index.tolist() == [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]
        assert edge_attr.tolist() == [1., 1., 1., 1., 1., 1.]


@withCUDA
@pytest.mark.parametrize(
    'layout',
    [torch.sparse_coo, torch.sparse_csr, torch.sparse_csc],
)
@pytest.mark.parametrize('dim', [0, 1, (0, 1)])
def test_cat(layout, dim, device):
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], device=device)
    if torch_geometric.typing.WITH_PT20:
        edge_weight = torch.rand(4, 2, device=device)
    else:
        edge_weight = torch.rand(4, device=device)

    adj = to_torch_sparse_tensor(edge_index, edge_weight, layout=layout)

    out = cat([adj, adj], dim=dim)
    edge_index, edge_weight = to_edge_index(out.to_sparse_csr())

    if dim == 0:
        if torch_geometric.typing.WITH_PT20:
            assert out.size() == (6, 3, 2)
        else:
            assert out.size() == (6, 3)
        assert edge_index[0].tolist() == [0, 1, 1, 2, 3, 4, 4, 5]
        assert edge_index[1].tolist() == [1, 0, 2, 1, 1, 0, 2, 1]
    elif dim == 1:
        if torch_geometric.typing.WITH_PT20:
            assert out.size() == (3, 6, 2)
        else:
            assert out.size() == (3, 6)
        assert edge_index[0].tolist() == [0, 0, 1, 1, 1, 1, 2, 2]
        assert edge_index[1].tolist() == [1, 4, 0, 2, 3, 5, 1, 4]
    else:
        if torch_geometric.typing.WITH_PT20:
            assert out.size() == (6, 6, 2)
        else:
            assert out.size() == (6, 6)
        assert edge_index[0].tolist() == [0, 1, 1, 2, 3, 4, 4, 5]
        assert edge_index[1].tolist() == [1, 0, 2, 1, 4, 3, 5, 4]


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    num_nodes, num_edges = 10_000, 200_000
    edge_index = torch.randint(num_nodes, (2, num_edges), device=args.device)

    benchmark(
        funcs=[
            SparseTensor.from_edge_index, to_torch_coo_tensor,
            to_torch_csr_tensor, to_torch_csc_tensor
        ],
        func_names=['SparseTensor', 'To COO', 'To CSR', 'To CSC'],
        args=(edge_index, None, (num_nodes, num_nodes)),
        num_steps=50 if args.device == 'cpu' else 500,
        num_warmups=10 if args.device == 'cpu' else 100,
    )
