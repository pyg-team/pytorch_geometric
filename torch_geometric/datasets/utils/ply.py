import torch
import numpy as np
from plyfile import PlyData, make2d


def read_ply(path):
    # Read ply dataformat file.
    with open(path, 'rb') as f:
        plydata = PlyData.read(f)

    # Extract numpy arrays for vertices and faces.
    x = plydata['vertex']['x']
    y = plydata['vertex']['y']
    z = plydata['vertex']['z']
    vertices = np.stack((x, y, z), axis=1)

    faces = make2d(plydata['face']['vertex_indices'])
    faces = faces.astype(np.int64)

    # Conver to torch tensors.
    vertices = torch.FloatTensor(vertices)
    faces = torch.LongTensor(faces)

    return vertices, faces
