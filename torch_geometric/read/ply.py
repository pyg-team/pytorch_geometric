import torch
from plyfile import PlyData
from torch_geometric.data import Data


def read_ply(path):
    with open(path, 'rb') as f:
        data = PlyData.read(f)

    pos = ([torch.tensor(data['vertex'][axis]) for axis in ['x', 'y', 'z']])
    pos = torch.stack(pos, dim=-1)

    faces = data['face']['vertex_indices']
    faces = [torch.tensor(face, dtype=torch.long) for face in faces]
    face = torch.stack(faces, dim=-1)

    data = Data(pos=pos)
    data.face = face

    return data
