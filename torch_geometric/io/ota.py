import sys
import os.path as osp
from itertools import repeat

import numpy as np
import torch
from torch_geometric.utils import from_scipy_sparse_matrix
from scipy.sparse import csr_matrix, block_diag

from torch_sparse import coalesce as coalesce_fn, SparseTensor
from torch_geometric.data import Data
from torch_geometric.io import read_txt_array
from torch_geometric.utils import remove_self_loops
import pickle


def read_ota_data(folder, name):
    print(folder)
    all_x = np.empty((0,16),np.int8)
    all_y = np.empty((0,1), int)
    all_designs = read_file(folder, name)
    adj = csr_matrix((0,0))
    for circuit_name, circuit_data in all_designs.items():
        df = circuit_data["data_matrix"]
        # print(circuit_name)
        node_features = df.values
        # print(node_features, type(node_features))
        node_features = np.delete(node_features, 0, 1)
        node_features = np.array(node_features, dtype=np.int8)
        node_features = node_features[:, 0:16]
        all_x = np.append(all_x, node_features, axis=0)
        all_y = np.append(all_y, circuit_data["target"])
        adj = block_diag((adj, circuit_data["adjacency_matrix"]))
        # print(adj.todense())
        # print(edge_index, edge_weight)
    edge_index, edge_weight = from_scipy_sparse_matrix(adj)
    x = torch.Tensor(all_x)
    y = torch.tensor(all_y.astype(np.int_))
    print(x)
    print(type(x))
    print(y.size())
    # x, tx, allx, y, ty, ally, graph, test_index = items
    train_index = torch.arange(y.size(0)*0.6, dtype=torch.long)
    val_index = torch.arange(y.size(0)*0.8, y.size(0), dtype=torch.long)
    test_index = torch.arange(y.size(0)*0.8, y.size(0), dtype=torch.long)
    train_mask = index_to_mask(train_index, size=y.size(0))
    val_mask = index_to_mask(val_index, size=y.size(0))
    test_mask = index_to_mask(test_index, size=y.size(0))

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data


def read_file(folder, name):
    path = osp.join(folder, name)
    print(path)

    with open(path, 'rb') as f:
        out = pickle.load(f)

    # out = out.todense() if hasattr(out, 'todense') else out
    # out = torch.Tensor(out)
    return out


def edge_index_from_dict(graph_dict, num_nodes=None, coalesce=True):
    row, col = [], []
    for key, value in graph_dict.items():
        row += repeat(key, len(value))
        col += value
    edge_index = torch.stack([torch.tensor(row), torch.tensor(col)], dim=0)
    if coalesce:
        # NOTE: There are some duplicated edges and self loops in the datasets.
        #       Other implementations do not remove them!
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = coalesce_fn(edge_index, None, num_nodes, num_nodes)
    return edge_index


def index_to_mask(index, size):
    mask = torch.zeros((size, ), dtype=torch.bool)
    mask[index] = 1
    return mask
