from itertools import repeat
import os
import sys

import torch

try:
    import cPickle as pickle
except ImportError:
    import pickle

from .progress import Progress


def read_planetoid(dir, name):
    progress = Progress('Processing', '', end=5, type='')

    # Read input files.
    objects = []
    for e in ['tx', 'ty', 'allx', 'ally', 'graph']:
        file_name = os.path.join(dir, 'ind.{}.{}'.format(name, e))
        with open(file_name, 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pickle.load(f, encoding='latin1'))
            else:
                objects.append(pickle.load(f))

    tx, ty, allx, ally, graph = tuple(objects)
    progress.update(1)

    with open(os.path.join(dir, 'ind.{}.test.index'.format(name)), 'r') as f:
        test_index = torch.LongTensor([int(line.strip()) for line in f])
    test_index_sorted = test_index.sort()[0]
    progress.update(2)

    # Input.
    tx = torch.FloatTensor(tx.todense())
    allx = torch.FloatTensor(allx.todense())
    input = torch.cat([allx, tx], dim=0)
    input[test_index, :] = input[test_index_sorted, :]
    progress.update(3)

    # Graph.
    row = []
    col = []
    for key, value in graph.items():
        row += repeat(key, len(value))
        col += value

    row, col = torch.LongTensor(row), torch.LongTensor(col)
    index = torch.stack([row, col], dim=0)
    progress.update(4)

    # Target.
    ty, ally = torch.IntTensor(ty), torch.IntTensor(ally)
    target = torch.cat([ally, ty], dim=0)
    target = target.max(dim=1)[1].byte()
    target[test_index] = target[test_index_sorted]
    progress.success()

    return input, index, target
