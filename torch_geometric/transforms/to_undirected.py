from typing import Union

import torch
from torch import Tensor

from torch_geometric.utils import to_undirected
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform


class ToUndirected(BaseTransform):
    r"""Converts a homogeneous or heterogeneous graph to an undirected graph
    such that :math:`(j,i) \in \mathcal{E}` for every edge
    :math:`(i,j) \in \mathcal{E}`.
    For bipartite connections in heterogeneous graphs, will only add "inverse"
    connections between two different node types in case there does not yet
    exist a single connection between these two node types.

    Args:
        reduce (string, optional): The reduce operation to use for merging edge
            features (:obj:`"add"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`,
            :obj:`"mul"`). (default: :obj:`"add"`)
    """
    def __init__(self, reduce: str = "add"):
        self.reduce = reduce

    def __call__(self, data: Union[Data, HeteroData]):
        pairs = set()
        if isinstance(data, HeteroData):  # Collect all connected node pairs:
            pairs = {(src, dst) for src, _, dst in data.edge_types}

        for store in data.edge_stores:
            if 'edge_index' not in store:
                continue

            nnz = store.edge_index.size(1)

            if store.is_bipartite():
                src, rel, dst = store._key
                if (dst, src) in pairs:
                    continue

                # Just inverse the connectivity and add edge attributes:
                row, col = store.edge_index
                edge_index = torch.stack([col, row], dim=0)

                if rel != HeteroData.DEFAULT_REL:
                    rel = f'rev_{rel}'
                inv_store = data[dst, rel, src]
                inv_store.edge_index = edge_index
                for key, value in store.items():
                    if isinstance(value, Tensor) and value.size(0) == nnz:
                        inv_store[key] = value

            else:
                keys, values = [], []
                for key, value in store.items():
                    if isinstance(value, Tensor) and value.size(0) == nnz:
                        keys.append(key)
                        values.append(value)

                store.edge_index, values = to_undirected(
                    store.edge_index, values, reduce=self.reduce)

                for key, value in zip(keys, values):
                    store[key] = value

        return data

    def __repr__(self):
        return f'{self.__class__.__name__}()'
