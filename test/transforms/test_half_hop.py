import torch

from torch_geometric.data import Data
from torch_geometric.transforms import HalfHop


def test_half_hops():
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    x = torch.tensor(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
        dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)

    transform = HalfHop()
    assert str(transform) == 'HalfHop(alpha=0.5, p=1)'
    data = transform(data)

    expected_edge_index = [[
        0, 1, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 1, 0, 2, 1, 3, 2
    ], [4, 5, 6, 7, 8, 9, 1, 0, 2, 1, 3, 2, 4, 5, 6, 7, 8, 9]]
    expected_x = torch.tensor([[1., 2., 3., 4.], [5., 6., 7., 8.],
                               [9., 10., 11., 12.], [13., 14., 15., 16.],
                               [3., 4., 5., 6.], [3., 4., 5., 6.],
                               [7., 8., 9., 10.], [7., 8., 9., 10.],
                               [11., 12., 13., 14.], [11., 12., 13., 14.]])
    assert data.num_nodes == 10
    assert data.edge_index.tolist() == expected_edge_index
    assert torch.allclose(data.x, expected_x, atol=1e-4)
