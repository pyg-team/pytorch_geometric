import torch

from ...graph.geometry import edges_from_faces
from ..dataset import Data


def read_off(filename):
    with open(filename, 'r') as f:
        data = f.read().split()

    num_nodes = int(data[1])

    pos = [float(x) for x in data[4:num_nodes * 3 + 4]]
    pos = torch.FloatTensor(pos).view(-1, 3)

    face = [int(x) for x in data[num_nodes * 3 + 4:]]
    face = torch.LongTensor(face).view(-1, 4)[:, 1:]

    index = edges_from_faces(face)

    return Data(None, pos, index, None, None)
