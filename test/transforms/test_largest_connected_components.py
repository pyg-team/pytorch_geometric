import torch

from torch_geometric.data import Data
from torch_geometric.transforms import LargestConnectedComponents


def test_largest_connected_components():
    assert str(LargestConnectedComponents()) == 'LargestConnectedComponents(1)'

    edge_index = torch.tensor([
        [0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6, 8, 9],
        [1, 2, 0, 2, 0, 1, 3, 2, 4, 3, 6, 7, 9, 8],
    ])
    data = Data(edge_index=edge_index, num_nodes=10)

    transform = LargestConnectedComponents(num_components=2)
    out = transform(data)
    assert out.num_nodes == 8
    assert out.edge_index.tolist() == data.edge_index[:, :12].tolist()

    edge_index = torch.tensor([
        [0, 1, 2, 3, 3, 4],
        [1, 0, 3, 2, 4, 3],
    ])
    data = Data(edge_index=edge_index, num_nodes=5)

    transform = LargestConnectedComponents()
    out = transform(data)
    assert out.num_nodes == 3
    assert out.edge_index.tolist() == [[0, 1, 1, 2], [1, 0, 2, 1]]
