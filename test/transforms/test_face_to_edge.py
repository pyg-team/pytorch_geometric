import torch
from torch_geometric.transforms import FaceToEdge
from torch_geometric.data import Data


def test_face_to_edge():
    assert FaceToEdge().__repr__() == 'FaceToEdge()'

    face = torch.tensor([[0, 0], [1, 1], [2, 3]])

    data = Data()
    data.face = face
    data = FaceToEdge()(data)
    assert len(data) == 1
    assert data.edge_index.tolist() == [[0, 0, 0, 1, 1, 1, 2, 2, 3, 3],
                                        [1, 2, 3, 0, 2, 3, 0, 1, 0, 1]]
