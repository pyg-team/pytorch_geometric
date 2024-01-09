import os.path as osp
import warnings
from itertools import repeat
from typing import Dict, List, Optional

import fsspec
import torch
from torch import Tensor

from torch_geometric.data import Data
from torch_geometric.io import read_txt_array
from torch_geometric.utils import (
    coalesce,
    index_to_mask,
    remove_self_loops,
    to_torch_csr_tensor,
)

try:
    import cPickle as pickle
except ImportError:
    import pickle


def read_planetoid_data(folder: str, prefix: str) -> Data:
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
        len_test_indices = int(test_index.max() - test_index.min()) + 1

        tx_ext = torch.zeros(len_test_indices, tx.size(1), dtype=tx.dtype)
        tx_ext[sorted_test_index - test_index.min(), :] = tx
        ty_ext = torch.zeros(len_test_indices, ty.size(1), dtype=ty.dtype)
        ty_ext[sorted_test_index - test_index.min(), :] = ty

        tx, ty = tx_ext, ty_ext

    if prefix.lower() == 'nell.0.001':
        tx_ext = torch.zeros(len(graph) - allx.size(0), x.size(1))
        tx_ext[sorted_test_index - allx.size(0)] = tx

        ty_ext = torch.zeros(len(graph) - ally.size(0), y.size(1))
        ty_ext[sorted_test_index - ally.size(0)] = ty

        tx, ty = tx_ext, ty_ext

        x = torch.cat([allx, tx], dim=0)
        x[test_index] = x[sorted_test_index]

        # Creating feature vectors for relations.
        row, col = x.nonzero(as_tuple=True)
        value = x[row, col]

        mask = ~index_to_mask(test_index, size=len(graph))
        mask[:allx.size(0)] = False
        isolated_idx = mask.nonzero().view(-1)

        row = torch.cat([row, isolated_idx])
        col = torch.cat([col, torch.arange(isolated_idx.size(0)) + x.size(1)])
        value = torch.cat([value, value.new_ones(isolated_idx.size(0))])

        x = to_torch_csr_tensor(
            edge_index=torch.stack([row, col], dim=0),
            edge_attr=value,
            size=(x.size(0), isolated_idx.size(0) + x.size(1)),
        )
    else:
        x = torch.cat([allx, tx], dim=0)
        x[test_index] = x[sorted_test_index]

    y = torch.cat([ally, ty], dim=0).max(dim=1)[1]
    y[test_index] = y[sorted_test_index]

    train_mask = index_to_mask(train_index, size=y.size(0))
    val_mask = index_to_mask(val_index, size=y.size(0))
    test_mask = index_to_mask(test_index, size=y.size(0))

    edge_index = edge_index_from_dict(
        graph_dict=graph,  # type: ignore
        num_nodes=y.size(0),
    )

    data = Data(x=x, edge_index=edge_index, y=y)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data


def read_file(folder: str, prefix: str, name: str) -> Tensor:
    path = osp.join(folder, f'ind.{prefix.lower()}.{name}')

    if name == 'test.index':
        return read_txt_array(path, dtype=torch.long)

    with fsspec.open(path, 'rb') as f:
        warnings.filterwarnings('ignore', '.*`scipy.sparse.csr` name.*')
        out = pickle.load(f, encoding='latin1')

    if name == 'graph':
        return out

    out = out.todense() if hasattr(out, 'todense') else out
    out = torch.from_numpy(out).to(torch.float)
    return out


def edge_index_from_dict(
    graph_dict: Dict[int, List[int]],
    num_nodes: Optional[int] = None,
) -> Tensor:
    rows: List[int] = []
    cols: List[int] = []
    for key, value in graph_dict.items():
        rows += repeat(key, len(value))
        cols += value
    row = torch.tensor(rows)
    col = torch.tensor(cols)
    edge_index = torch.stack([row, col], dim=0)

    # `torch.compile` is not yet ready for `EdgeIndex` :(
    # from torch_geometric import EdgeIndex
    # edge_index: Union[EdgeIndex, Tensor] = EdgeIndex(
    #     torch.stack([row, col], dim=0),
    #     is_undirected=True,
    #     sparse_size=(num_nodes, num_nodes),
    # )

    # NOTE: There are some duplicated edges and self loops in the datasets.
    #       Other implementations do not remove them!
    edge_index, _ = remove_self_loops(edge_index)
    edge_index = coalesce(edge_index, num_nodes=num_nodes, sort_by_row=False)

    return edge_index
