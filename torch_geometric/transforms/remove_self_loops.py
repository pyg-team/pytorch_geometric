from typing import Union

from torch_geometric.data import Data, HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import remove_self_loops


@functional_transform('remove_self_loops')
class RemoveSelfLoops(BaseTransform):
    r"""Removes all self-loops in the given homogeneous or heterogeneous
    graph (functional name: :obj:`remove_self_loops`).

    Args:
        attr (str, optional): The name of the attribute of edge weights
            or multi-dimensional edge features to pass to
            :meth:`torch_geometric.utils.remove_self_loops`.
            (default: :obj:`"edge_weight"`)
    """
    def __init__(self, attr: str = 'edge_weight') -> None:
        self.attr = attr

    def forward(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        for store in data.edge_stores:
            if store.is_bipartite() or 'edge_index' not in store:
                continue

            store.edge_index, store[self.attr] = remove_self_loops(
                store.edge_index,
                edge_attr=store.get(self.attr, None),
            )

        return data
