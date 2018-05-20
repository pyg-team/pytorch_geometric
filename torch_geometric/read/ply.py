import torch
from plyfile import PlyData
from torch_geometric.data import Data
from torch_geometric.utils import face_to_edge_index


def read_ply_data(path):
    with open(path, 'rb') as f:
        data = PlyData.read(f)

    pos = ([torch.tensor(data['vertex'][axis]) for axis in ['x', 'y', 'z']])
    pos = torch.stack(pos, dim=-1)

    faces = data['face']['vertex_indices']
    faces = [torch.tensor(face, dtype=torch.long) for face in faces]
    face = torch.stack(faces, dim=-1)

    edge_index = face_to_edge_index(face, num_nodes=pos.size(0))

    data = Data(edge_index=edge_index, pos=pos)
    data.face = face

    return data
