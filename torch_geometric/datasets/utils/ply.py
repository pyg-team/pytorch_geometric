import torch
from plyfile import PlyData, make2d

from ...graph.geometry import edges_from_faces
from ..dataset import Data


def read_ply(filename):
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)

    x = torch.FloatTensor(plydata['vertex']['x'])
    y = torch.FloatTensor(plydata['vertex']['y'])
    z = torch.FloatTensor(plydata['vertex']['z'])
    pos = torch.stack([x, y, z], dim=1)
    input = torch.ones(pos.size(0))

    face = torch.LongTensor(make2d(plydata['face']['vertex_indices']))
    index = edges_from_faces(face)

    return Data(input, pos, index, None, None)
