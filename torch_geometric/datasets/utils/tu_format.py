import os

import torch


def read_file(dir, prefix, name):
    path = os.path.join(dir, '{}_{}.txt'.format(prefix, name))
    with open(path, 'r') as f:
        lines = f.read().split('\n')[:-1]
        content = [[float(x) for x in line.split(', ')] for line in lines]
    return torch.FloatTensor(content)
