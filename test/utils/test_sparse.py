import torch

from torch_geometric.testing import is_full_test
from torch_geometric.utils import dense_to_sparse


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
