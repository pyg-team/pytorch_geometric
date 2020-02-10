import torch
import openmesh
from torch_geometric.data import Data


def read_ply(path):
    mesh = openmesh.read_trimesh(path)
    pos = torch.from_numpy(mesh.points()).to(torch.float)
    face = torch.from_numpy(mesh.face_vertex_indices())
    face = face.t().to(torch.long).contiguous()
    data = Data(pos=pos, face=face)
    return data
