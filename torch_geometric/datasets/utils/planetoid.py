from itertools import repeat
import sys

import torch
try:
    import cPickle as pickle
except ImportError:
    import pickle

from ..dataset import Data


def pickle_load(filename):
    with open(filename, 'rb') as f:
        if sys.version_info > (3, 0):
            return pickle.load(f, encoding='latin1')
        else:
            return pickle.load(f)


def read_planetoid(filenames):
    # filename order = ['tx', 'ty', 'allx', 'ally', 'graph', 'test.index']
    tx, ty, allx, ally, graph = tuple([pickle_load(f) for f in filenames[:-1]])
    with open(filenames[-1], 'r') as f:
        test_index = torch.LongTensor([int(line.strip()) for line in f])
        test_index_sorted, _ = test_index.sort()

    # Fix datasets with isoloated nodes in the graph.
    # Find isolated nodes and add them as zero-vec into the right position.
    min, max = test_index_sorted[0], test_index_sorted[-1]
    if max - min > 1000:
        pass

    allx, tx = torch.Tensor(allx.todense()), torch.Tensor(tx.todense())
    input = torch.cat([allx, tx], dim=0)
    input[test_index, :] = input[test_index_sorted, :]

    ty, ally = torch.LongTensor(ty), torch.LongTensor(ally)
    target = torch.cat([ally, ty], dim=0)
    target[test_index, :] = target[test_index_sorted, :]
    target = target.max(dim=1)[1]

    row, col = [], []
    for key, value in graph.items():
        row += repeat(key, len(value))
        col += value
    row, col = torch.LongTensor(row), torch.LongTensor(col)
    index = torch.stack([row, col], dim=0)

    return Data(input, None, index, None, target)
