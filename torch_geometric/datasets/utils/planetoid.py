from itertools import repeat
import os
import sys

import torch

try:
    import cPickle as pickle
except ImportError:
    import pickle


def read_planetoid(dir, name):
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

    # Input.
    tx = torch.FloatTensor(tx.todense())
    allx = torch.FloatTensor(allx.todense())
    input = torch.cat([allx, tx], dim=0)

    # Graph.
    row = []
    col = []
    for key, value in graph.items():
        row += repeat(key, len(value))
        col += value
    row, col = torch.LongTensor(row), torch.LongTensor(col)
    index = torch.stack([row, col], dim=0)

    # Target.
    ty = torch.IntTensor(ty)
    ally = torch.IntTensor(ally)
    target = torch.cat([ally, ty], dim=0)
    target = target.max(dim=1)[1].byte()

    return input, index, target
