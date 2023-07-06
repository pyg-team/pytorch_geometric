import torch

from torch_geometric.data import Data
from torch_geometric.transforms import GenerateMeshNormals


def test_generate_mesh_normals():
    transform = GenerateMeshNormals()
    assert str(transform) == 'GenerateMeshNormals()'

    pos = torch.tensor([
        [0.0, 0.0, 0.0],
        [-2.0, 1.0, 0.0],
        [-1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [2.0, 1.0, 0.0],
    ])
    face = torch.tensor([
        [0, 0, 0, 0],
        [1, 2, 3, 4],
        [2, 3, 4, 5],
    ])

    data = transform(Data(pos=pos, face=face))
    assert len(data) == 3
    assert data.pos.tolist() == pos.tolist()
    assert data.face.tolist() == face.tolist()
    assert data.norm.tolist() == [[0.0, 0.0, -1.0]] * 6
