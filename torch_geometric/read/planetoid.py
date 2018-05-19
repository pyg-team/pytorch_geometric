import sys
import os.path as osp
from itertools import repeat

import torch
from torch_geometric.utils import coalesce
from torch_geometric.data import Data
from torch_geometric.read import read_txt_array

try:
    import cPickle as pickle
except ImportError:
    import pickle


def read_planetoid_data(folder, prefix):
    """Reads the planetoid data format.
    ind.{}.x
    """
    names = ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']
    items = [read_file(folder, prefix, name) for name in names]
    x, tx, allx, y, ty, ally, graph, test_index = items
    train_index = torch.arange(y.size(0), dtype=torch.long)
    val_index = torch.arange(y.size(0), y.size(0) + 500, dtype=torch.long)

    if prefix.lower() == 'citeseer':
        pass

    x = torch.cat([allx, tx], dim=0)
    y = torch.cat([ally, ty], dim=0).max(dim=1)[1]

    sorted_test_index = test_index.sort()[0]
    x[test_index] = x[sorted_test_index]
    y[test_index] = y[sorted_test_index]

    train_mask = sample_mask(train_index, num_nodes=y.size(0))
    val_mask = sample_mask(val_index, num_nodes=y.size(0))
    test_mask = sample_mask(test_index, num_nodes=y.size(0))

    edge_index = edge_index_from_dict(graph, num_nodes=y.size(0))

    data = Data(x=x, edge_index=edge_index, y=y)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data


def read_file(folder, prefix, name):
    path = osp.join(folder, 'ind.{}.{}'.format(prefix.lower(), name))

    if name == 'test.index':
        obj = read_txt_array(path, dtype=torch.long)
        print(obj.size())
        return obj

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


def edge_index_from_dict(graph_dict, num_nodes=None):
    row, col = [], []
    for key, value in graph_dict.items():
        row += repeat(key, len(value))
        col += value
    edge_index = torch.stack([torch.tensor(row), torch.tensor(col)], dim=0)
    # NOTE: There are duplicated edges in the planetoid datasets. Other
    # implementations do not remove them.
    edge_index, _ = coalesce(edge_index, num_nodes)
    return edge_index


def sample_mask(index, num_nodes):
    mask = torch.zeros((num_nodes, ), dtype=torch.uint8)
    mask[index] = 1
    return mask
