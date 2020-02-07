import torch
import numpy as np
import scipy.sparse as sp
from torch_sparse import SparseTensor
from torch_geometric.data import Data


def read_npz(path):
    with np.load(path) as f:
        return parse_npz(f)


def parse_npz(f):
    adj = sp.csr_matrix((f['adj_data'], f['adj_indices'], f['adj_indptr']),
                        f['adj_shape'])
    adj = SparseTensor.from_scipy(adj, has_value=False)
    adj = adj.remove_diag().to_symmetric().fil_cache_()

    x = sp.csr_matrix((f['attr_data'], f['attr_indices'], f['attr_indptr']),
                      f['attr_shape']).todense()
    x = torch.from_numpy(x).to(torch.float)
    x[x > 0] = 1

    y = torch.from_numpy(f['labels']).to(torch.long)

    return Data(adj=adj, x=x, y=y)
