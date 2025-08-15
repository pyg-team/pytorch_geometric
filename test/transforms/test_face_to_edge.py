import torch

from torch_geometric.data import Data
from torch_geometric.transforms import FaceToEdge


def test_2d_face_to_edge() -> None:
    transform = FaceToEdge()
    assert str(transform) == 'FaceToEdge()'

    face = torch.tensor([[0, 0], [1, 1], [2, 3]])
    data = Data(face=face, num_nodes=4)

    data = transform(data)
    assert len(data) == 2
    assert data.edge_index.tolist() == [
        [0, 0, 0, 1, 1, 1, 2, 2, 3, 3],
        [1, 2, 3, 0, 2, 3, 0, 1, 0, 1],
    ]
    assert data.num_nodes == 4


def test_3d_face_to_edge() -> None:
    transform = FaceToEdge()
    assert str(transform) == 'FaceToEdge()'

    face = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]).t()
    data = Data(face=face, num_nodes=5)

    data = transform(data)
    assert data.edge_index.tolist() == [
        [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4],
        [1, 2, 3, 0, 2, 3, 4, 0, 1, 3, 4, 0, 1, 2, 4, 1, 2, 3],
    ]
    assert data.num_nodes == 5
