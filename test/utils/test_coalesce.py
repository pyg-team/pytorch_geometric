import torch
from torch_geometric.utils import coalesce


def test_coalesce():
    row = torch.tensor([1, 0, 1, 0, 2, 1])
    col = torch.tensor([0, 1, 1, 1, 0, 0])
    edge_index = torch.stack([row, col], dim=0)
    edge_attr = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]]
    edge_attr = torch.tensor(edge_attr)

    out, _ = coalesce(edge_index)
    assert out.tolist() == [[0, 1, 1, 2], [1, 0, 1, 0]]

    out = coalesce(edge_index, edge_attr)
    assert out[0].tolist() == [[0, 1, 1, 2], [1, 0, 1, 0]]
    assert out[1].tolist() == [[10, 12], [12, 14], [5, 6], [9, 10]]
