from typing import List, Optional, Union

from torch_geometric.data import Data, HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import coalesce


@functional_transform('remove_duplicated_edges')
class RemoveDuplicatedEdges(BaseTransform):
    r"""Removes duplicated edges from a given homogeneous or heterogeneous
    graph. Useful to clean-up known repeated edges/self-loops in common
    benchmark datasets, *e.g.*, in :obj:`ogbn-products`.
    (functional name: :obj:`remove_duplicated_edges`).

    Args:
        key (str or [str], optional): The name of edge attribute(s) to merge in
            case of duplication. (default: :obj:`["edge_weight", "edge_attr"]`)
        reduce (str, optional): The reduce operation to use for merging edge
            attributes (:obj:`"add"`, :obj:`"mean"`, :obj:`"min"`,
            :obj:`"max"`, :obj:`"mul"`). (default: :obj:`"add"`)
    """
    def __init__(
        self,
        key: Optional[Union[str, List[str]]] = None,
        reduce: str = "add",
    ) -> None:
        key = key or ['edge_attr', 'edge_weight']

        if isinstance(key, str):
            key = [key]

        self.keys = key
        self.reduce = reduce

    def forward(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:

        for store in data.edge_stores:
            keys = [key for key in self.keys if key in store]

            size = [s for s in store.size() if s is not None]
            num_nodes = max(size) if len(size) > 0 else None

            store.edge_index, edge_attrs = coalesce(
                edge_index=store.edge_index,
                edge_attr=[store[key] for key in keys],
                num_nodes=num_nodes,
                reduce=self.reduce,
            )

            for key, edge_attr in zip(keys, edge_attrs):
                store[key] = edge_attr

        return data
