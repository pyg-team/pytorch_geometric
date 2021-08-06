import torch
from torch_geometric.transforms import TetraToEdge
from torch_geometric.data import Data


def test_tetra_to_edge():
    assert TetraToEdge().__repr__() == 'TetraToEdge()'

    tetra = torch.tensor([[0, 0], [1, 2], [2, 3], [3, 4]])

    data = Data(tetra=tetra, num_nodes=5)
    data = TetraToEdge()(data)
    assert len(data) == 1
    assert data.edge_index.tolist() == [
            [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4],
            [1, 2, 3, 4, 0, 2, 3, 0, 1, 3, 4, 0, 1, 2, 4, 0, 2, 3]
        ]
