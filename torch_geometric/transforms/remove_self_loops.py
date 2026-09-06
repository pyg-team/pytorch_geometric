from typing import List, Union

from torch_geometric.data import Data, HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import remove_self_loops


@functional_transform('remove_self_loops')
class RemoveSelfLoops(BaseTransform):
    r"""Removes all self-loops in the given homogeneous or heterogeneous
    graph (functional name: :obj:`remove_self_loops`).

    Args:
        attr (str or [str], optional): The name of the edge attribute(s) of
            edge weights or multi-dimensional edge features to pass to
            :meth:`torch_geometric.utils.remove_self_loops` and keep in sync
            with the filtered :obj:`edge_index`. Pass a list to keep
            additional edge-level attributes (*e.g.*, :obj:`"time"` in a
            temporal graph) aligned with the filtered edges as well.
            (default: :obj:`"edge_weight"`)
    """
    def __init__(self, attr: Union[str, List[str]] = 'edge_weight') -> None:
        self.attrs = [attr] if isinstance(attr, str) else attr

    def forward(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        for store in data.edge_stores:
            if store.is_bipartite() or 'edge_index' not in store:
                continue

            # Filter every attribute against the *original* `edge_index`, not
            # a previously filtered one, so each attribute's self-loop mask
            # lines up with its own (still full-length) tensor.
            edge_index = store.edge_index
            for attr in self.attrs:
                edge_index, store[attr] = remove_self_loops(
                    store.edge_index,
                    edge_attr=store.get(attr, None),
                )
            store.edge_index = edge_index

        return data
