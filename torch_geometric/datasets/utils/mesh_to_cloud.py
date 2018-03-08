import numpy as np
import torch

from ..dataset import Data
from .nn_graph import nn_graph

def generate_cloud(filename):

    with open(filename, 'r') as f:
        data = f.read().split()

    if data[0] == 'OFF':
        data.pop(0)
    else:
        # Some files contain a bug and don't have a carriage return after OFF.
        data[0] = data[0][3:]

    num_nodes = int(data[0])

    pos = [float(x) for x in data[3:num_nodes * 3 + 3]]
    pos = torch.FloatTensor(pos).view(-1, 3)

    face = [int(x) for x in data[num_nodes * 3 + 3:]]
    face = torch.LongTensor(face).view(-1, 4)[:, 1:].contiguous()

    num_points = 2**11

    # Create input features.
    input = torch.ones(pos.size(0))



    triangles = pos[face.view(-1)].view(-1,3,3)


    probs = np.sqrt((np.cross(triangles[:, 1, :] - triangles[:, 0, :],
                              triangles[:, 2, :] - triangles[:, 0,
                                                   :]) ** 2).sum(
        axis=1)) / 2.
    probs /= probs.sum()

    sample = torch.LongTensor(np.random.choice(np.arange(len(triangles)),
                                               size=num_points, p=probs))

    samples_0 = np.random.random((num_points, 1))
    samples_1 = np.random.random((num_points, 1))
    cond = (samples_0[:, 0] + samples_1[:, 0]) > 1.
    samples_0[cond, 0] = 1. - samples_0[cond, 0]
    samples_1[cond, 0] = 1. - samples_1[cond, 0]

    samples_0 = torch.FloatTensor(samples_0)
    samples_1 = torch.FloatTensor(samples_1)

    tri_points0 = triangles[sample][:, 0, :]
    tri_points1 = triangles[sample][:, 1, :]
    tri_points2 = triangles[sample][:, 2, :]

    pos = tri_points0 + samples_0 * (tri_points1 - tri_points0) + \
            samples_1 * (tri_points2 - tri_points0)

    input = torch.ones(pos.size(0))
    index = nn_graph(pos, k=6)

    return Data(input, pos, index, None, None)

def load_triangles(filename):

    with open(filename, 'r') as f:
        data = f.read().split()

    if data[0] == 'OFF':
        data.pop(0)
    else:
        # Some files contain a bug and don't have a carriage return after OFF.
        data[0] = data[0][3:]

    num_nodes = int(data[0])

    pos = [float(x) for x in data[3:num_nodes * 3 + 3]]
    pos = torch.FloatTensor(pos).view(-1, 3)

    face = [int(x) for x in data[num_nodes * 3 + 3:]]
    face = torch.LongTensor(face).view(-1, 4)[:, 1:].contiguous()

    triangles = pos[face.view(-1)].view(-1,3,3)

    return Data(None, None, None, None, None, faces=triangles)


#generate_cloud('/home/jan/PycharmProjects/pytorch_geometric/data/ModelNet10/raw/ModelNet10/bed/train/bed_0001.off')