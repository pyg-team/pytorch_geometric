from typing import Union, Optional

from torch_geometric.utils import add_self_loops
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform


class AddSelfLoops(BaseTransform):
    r"""Adds self-loops to the given homogeneous or heterogeneous graph.

    Args:
        attr: (str, optional): The name of the attribute of edge weights
            or multi-dimensional edge features, to pass it to the
            :meth:`torch_geometric.utils.add_self_loops`.
            (default: :obj:`edge_weight`)
    """
    def __init__(self, attr: Optional[str] = 'edge_weight'):
        self.attr = attr

    def __call__(self, data: Union[Data, HeteroData]):
        for store in data.edge_stores:
            if store.is_bipartite() or 'edge_index' not in store:
                continue

            store.edge_index, store.edge_weight = add_self_loops(
                store.edge_index,
                getattr(store, self.attr, None),
                num_nodes=store.size(0))

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
