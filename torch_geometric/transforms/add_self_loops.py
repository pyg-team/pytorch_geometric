from typing import Union

from torch_geometric.utils import add_self_loops
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform


class AddSelfLoops(BaseTransform):
    r"""Adds self-loops to the given homogeneous or heterogeneous graph."""
    def __call__(self, data: Union[Data, HeteroData]):
        for store in data.edge_stores:
            if store.is_bipartite() or 'edge_index' not in store:
                continue

            store.edge_index, store.edge_weight = add_self_loops(
                store.edge_index,
                store.edge_weight if 'edge_weight' in store else None,
                num_nodes=store.size(0))

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
