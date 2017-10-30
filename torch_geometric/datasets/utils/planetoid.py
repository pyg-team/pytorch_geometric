import os

import pickle
import torch


def read_planetoid(dir, name):
    ind = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in ind:
        file_name = os.path.join(dir, 'ind.{}.{}'.format(name, i))
        with open(file_name, 'rb') as f:
            objects.append(pickle.load(f, encoding='latin1'))

    x, y, tx, ty, allx, ally, graph = tuple(objects)

    # with open(os.path.join(dir, 'ind.{}.test.index'.format(name)), 'r') as f:
    #     test_index = torch.LongTensor([int(line.strip()) for line in f])
    # test_index_sorted, _ = test_index.sort()

    allx = torch.FloatTensor(allx.todense())
    tx = torch.FloatTensor(tx.todense())
    input = torch.cat([allx, tx], dim=0)
    # input[test_index, :] = input[test_index_sorted, :]

    ally = torch.IntTensor(ally).long()
    ty = torch.IntTensor(ty).long()
    target = torch.cat([ally, ty], dim=0)
    _, target = target.max(dim=1)
    # target[test_index, :] = target[test_index_sorted, :]

    row = []
    col = []
    for key, value in graph.items():
        row.extend([key for _ in range(len(value))])
        col.extend(value)

    row, col = torch.LongTensor(row), torch.LongTensor(col)
    index = torch.stack([row, col], dim=0)

    return input, index, target
