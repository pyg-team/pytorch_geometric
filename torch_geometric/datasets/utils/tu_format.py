import os

import torch
import numpy as np


def read_file(dir, prefix, name):
    path = os.path.join(dir, '{}_{}.txt'.format(prefix, name))
    with open(path, 'r') as f:
        lines = f.read().split('\n')[:-1]
        content = [[float(x) for x in line.split(', ')] for line in lines]
    return torch.FloatTensor(content)


def read_adj(dir, prefix):
    index = read_file(dir, prefix, 'A')
    index = index.t().long() - 1
    new_index = index.new(index.size()).copy_(index)

    indicator = read_file(dir, prefix, 'graph_indicator').squeeze().long() - 1
    bincount = torch.from_numpy(np.bincount(indicator.numpy()))

    index_slice = index.new(bincount.size(0) + 1)
    index_slice[0] = 0
    curr_graph = indicator[0]
    for i in range(index.size(1)):
        row = index[0, i]
        if indicator[row] > curr_graph:
            new_index[:, i:] -= bincount[curr_graph]
            curr_graph += 1
            index_slice[curr_graph] = i
    index_slice[-1] = index.size(1)
    index = new_index

    return index, index_slice
