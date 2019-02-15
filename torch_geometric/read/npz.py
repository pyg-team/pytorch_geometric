import torch
import numpy as np
import scipy.sparse as sp
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops, to_undirected


def read_npz(path):
    with np.load(path) as f:
        return parse_npz(f)


def parse_npz(f):
    x = sp.csr_matrix((f['attr_data'], f['attr_indices'], f['attr_indptr']),
                      f['attr_shape']).todense()
    x = torch.from_numpy(x).to(torch.float)
    x[x > 0] = 1

    adj = sp.csr_matrix((f['adj_data'], f['adj_indices'], f['adj_indptr']),
                        f['adj_shape']).tocoo()
    edge_index = torch.tensor([adj.row, adj.col], dtype=torch.long)
    edge_index, _ = remove_self_loops(edge_index)
    edge_index = to_undirected(edge_index, x.size(0))  # Internal coalesce.

    y = torch.from_numpy(f['labels']).to(torch.long)

    return Data(x=x, edge_index=edge_index, y=y)
