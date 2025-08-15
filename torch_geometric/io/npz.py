from typing import Any, Dict

import numpy as np
import torch

from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops
from torch_geometric.utils import to_undirected as to_undirected_fn


def read_npz(path: str, to_undirected: bool = True) -> Data:
    with np.load(path) as f:
        return parse_npz(f, to_undirected=to_undirected)


def parse_npz(f: Dict[str, Any], to_undirected: bool = True) -> Data:
    import scipy.sparse as sp

    x = sp.csr_matrix((f['attr_data'], f['attr_indices'], f['attr_indptr']),
                      f['attr_shape']).todense()
    x = torch.from_numpy(x).to(torch.float)
    x[x > 0] = 1

    adj = sp.csr_matrix((f['adj_data'], f['adj_indices'], f['adj_indptr']),
                        f['adj_shape']).tocoo()
    row = torch.from_numpy(adj.row).to(torch.long)
    col = torch.from_numpy(adj.col).to(torch.long)
    edge_index = torch.stack([row, col], dim=0)
    edge_index, _ = remove_self_loops(edge_index)
    if to_undirected:
        edge_index = to_undirected_fn(edge_index, num_nodes=x.size(0))

    y = torch.from_numpy(f['labels']).to(torch.long)

    return Data(x=x, edge_index=edge_index, y=y)
