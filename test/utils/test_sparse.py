import torch
from torch_geometric.utils import dense_to_sparse, sparse_to_dense


def test_dense_to_sparse():
    tensor = torch.Tensor([[3, 1], [2, 0]])
    edge_index, edge_attr = dense_to_sparse(tensor)

    assert edge_index.tolist() == [[0, 0, 1], [0, 1, 0]]
    assert edge_attr.tolist() == [3, 1, 2]


def test_sparse_to_dense():
    edge_index = torch.tensor([[0, 0, 1], [0, 1, 0]])
    edge_attr = torch.Tensor([3, 1, 2])

    tensor = sparse_to_dense(edge_index, edge_attr)
    assert tensor.tolist() == [[3, 1], [2, 0]]

    tensor = sparse_to_dense(edge_index)
    assert tensor.tolist() == [[1, 1], [1, 0]]
