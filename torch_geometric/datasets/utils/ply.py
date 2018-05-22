import torch
from plyfile import PlyData
import numpy as np

from ..dataset import Data
from ...utils.undirected import to_undirected


def read_ply(filename):
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
    x = torch.FloatTensor(plydata['vertex']['x'])
    y = torch.FloatTensor(plydata['vertex']['y'])
    z = torch.FloatTensor(plydata['vertex']['z'])
    pos = torch.stack([x, y, z], dim=1)
    input = torch.ones(pos.size(0))
    arrays = [np.expand_dims(v, 0) for v in plydata['face']['vertex_indices']]
    face = torch.LongTensor(np.concatenate(arrays, axis=0)).t()
    index = torch.cat([face[:2], face[1:], face[::2]], dim=1)
    index = to_undirected(index, num_nodes=pos.size(0))

    return Data(input, pos, index, None, None)
