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

    # Testing without `connection` specified:
    transform = LargestConnectedComponents(num_components=2)
    out = transform(data)
    assert out.num_nodes == 8
    assert out.edge_index.tolist() == data.edge_index[:, :12].tolist()

    # Testing with `connection = strong`:
    transform = LargestConnectedComponents(num_components=2,
                                           connection='strong')
    out = transform(data)
    assert out.num_nodes == 7
    assert out.edge_index.tolist() == [[0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6],
                                       [1, 2, 0, 2, 0, 1, 3, 2, 4, 3, 6, 5]]

    edge_index = torch.tensor([
        [0, 1, 2, 3, 3, 4],
        [1, 0, 3, 2, 4, 3],
    ])
    data = Data(edge_index=edge_index, num_nodes=5)

    # Testing without `num_components` and `connection` specified:
    transform = LargestConnectedComponents()
    out = transform(data)
    assert out.num_nodes == 3
    assert out.edge_index.tolist() == [[0, 1, 1, 2], [1, 0, 2, 1]]

    # Testing with larger `num_components` than actual number of components:
    transform = LargestConnectedComponents(num_components=3)
    out = transform(data)
    assert out.num_nodes == 5
    assert out.edge_index.tolist() == data.edge_index.tolist()
