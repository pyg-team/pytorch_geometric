import os

import torch
import numpy as np


def read_file(dir, prefix, name):
    path = os.path.join(dir, '{}_{}.txt'.format(prefix, name))
    with open(path, 'r') as f:
        lines = f.read().split('\n')[:-1]
        content = [[float(x) for x in line.split(', ')] for line in lines]
    return torch.FloatTensor(content).squeeze()


def read_adj(dir, prefix):
    index = read_file(dir, prefix, 'A')
    index = index.t().long() - 1
    row, col = index
    row, perm = row.sort()
    col = col[perm]
    index = torch.stack([row, col], dim=0)
    indicator = read_file(dir, prefix, 'graph_indicator').long() - 1

    index_slice = index.new(indicator.max() + 2)
    index_slice[0] = 0
    index_slice[-1] = index.size(1)
    curr_graph = indicator[0]

    for i in range(index.size(1)):
        row = index[0, i]
        if indicator[row] > curr_graph:
            j = index_slice[curr_graph]
            index[:, j:i] -= index[:, j:i].min()
            curr_graph += 1
            index_slice[curr_graph] = i
    index[:, index_slice[curr_graph]:] -= index[:, index_slice[
        curr_graph]:].min()

    return index, index_slice


def read_slice(dir, prefix):
    indicator = read_file(dir, prefix, 'graph_indicator').squeeze().long() - 1
    slice = np.cumsum(np.bincount(indicator.numpy()))
    slice = torch.from_numpy(slice)
    return torch.cat([torch.LongTensor([0]), slice], dim=0)
