import sys
import os.path as osp
from itertools import repeat

import torch
from torch_sparse import SparseTensor
from torch_geometric.data import Data
from torch_geometric.io import read_txt_array

try:
    import cPickle as pickle
except ImportError:
    import pickle


def read_planetoid_data(folder, prefix):
    names = ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']
    items = [read_file(folder, prefix, name) for name in names]
    x, tx, allx, y, ty, ally, graph, test_index = items
    train_index = torch.arange(y.size(0), dtype=torch.long)
    val_index = torch.arange(y.size(0), y.size(0) + 500, dtype=torch.long)
    sorted_test_index = test_index.sort()[0]

    if prefix.lower() == 'citeseer':
        # There are some isolated nodes in the Citeseer graph, resulting in
        # none consecutive test indices. We need to identify them and add them
        # as zero vectors to `tx` and `ty`.
        len_test_indices = (test_index.max() - test_index.min()).item() + 1

        tx_ext = torch.zeros(len_test_indices, tx.size(1))
        tx_ext[sorted_test_index - test_index.min(), :] = tx
        ty_ext = torch.zeros(len_test_indices, ty.size(1))
        ty_ext[sorted_test_index - test_index.min(), :] = ty

        tx, ty = tx_ext, ty_ext

    x = torch.cat([allx, tx], dim=0)
    y = torch.cat([ally, ty], dim=0).max(dim=1)[1]

    x[test_index] = x[sorted_test_index]
    y[test_index] = y[sorted_test_index]

    train_mask = index_to_mask(train_index, size=y.size(0))
    val_mask = index_to_mask(val_index, size=y.size(0))
    test_mask = index_to_mask(test_index, size=y.size(0))

    adj = adj_from_dict(graph, num_nodes=y.size(0))

    data = Data(adj=adj, x=x, y=y)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data


def read_file(folder, prefix, name):
    path = osp.join(folder, 'ind.{}.{}'.format(prefix.lower(), name))

    if name == 'test.index':
        return read_txt_array(path, dtype=torch.long)

    with open(path, 'rb') as f:
        if sys.version_info > (3, 0):
            out = pickle.load(f, encoding='latin1')
        else:
            out = pickle.load(f)

    if name == 'graph':
        return out

    out = out.todense() if hasattr(out, 'todense') else out
    out = torch.Tensor(out)
    return out


def adj_from_dict(graph_dict, num_nodes):
    row, col = [], []
    for key, value in graph_dict.items():
        row += repeat(key, len(value))
        col += value
    row, col = torch.tensor(row), torch.tensor(col)
    size = torch.Size([num_nodes, num_nodes])
    adj = SparseTensor(row=row, col=col, sparse_sizes=size)
    # NOTE: There are duplicated edges and self loops in the datasets. Other
    # implementations do not remove them!
    adj = adj.remove_diag().coalesce().fill_cache_()
    return adj


def index_to_mask(index, size):
    mask = torch.zeros((size, ), dtype=torch.bool)
    mask[index] = 1
    return mask
