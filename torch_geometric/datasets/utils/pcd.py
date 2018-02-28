import torch

from ..dataset import Data
from .nn_graph import nn_graph


def read_pcd(filename):
    with open(filename, 'r') as f:
        pos = f.read().split('\n')[11:-1]

    pos = torch.FloatTensor([[float(x) for x in line.split()] for line in pos])
    index = nn_graph(pos, k=6)
    input = torch.ones(pos.size(0))

    return Data(input, pos, index, None, None)
