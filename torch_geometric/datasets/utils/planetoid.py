import os
import sys

import torch

try:
    import cPickle as pickle
except ImportError:
    import pickle


def read_planetoid(dir, name):
    objects = []
    for e in ['tx', 'ty', 'allx', 'ally', 'graph']:
        file_name = os.path.join(dir, 'ind.{}.{}'.format(name, e))
        with open(file_name, 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pickle.load(f, encoding='latin1'))
            else:
                objects.append(pickle.load(f))

    tx, ty, allx, ally, graph = tuple(objects)

    with open(os.path.join(dir, 'ind.{}.test.index'.format(name)), 'r') as f:
        test_index = torch.LongTensor([int(line.strip()) for line in f])
    test_index_sorted, _ = test_index.sort()

    tx = torch.FloatTensor(tx.todense())
    ty = torch.IntTensor(ty).long()

    if name == 'citeseer':
        # Fix CiteSeer dataset (there are some isolated nodes in the graph).
        # Find isolated nodes, add them as zero-vecs into the right position.
        min_index, max_index = test_index.min(), test_index.max()
        len_full = max_index + 1 - min_index

        tx_extended = tx.new(len_full, tx.size(1)).fill_(0)
        tx_extended[test_index_sorted - min_index, :] = tx
        tx = tx_extended

        ty_extended = ty.new(len_full, ty.size(1)).fill_(0)
        ty_extended[test_index_sorted - min_index, :] = ty
        ty = ty_extended

    allx = torch.FloatTensor(allx.todense())
    input = torch.cat([allx, tx], dim=0)
    input[test_index, :] = input[test_index_sorted, :]

    ally = torch.IntTensor(ally).long()
    target = torch.cat([ally, ty], dim=0)
    target = target.max(dim=1)[1]
    target[test_index] = target[test_index_sorted]

    row = []
    col = []
    for key, value in graph.items():
        row.extend([key for _ in range(len(value))])

        col.extend(value)

    row, col = torch.LongTensor(row), torch.LongTensor(col)
    index = torch.stack([row, col], dim=0)

    return input, index, target, test_index_sorted
