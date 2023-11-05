from typing import List, Optional, Union

import torch

from torch_geometric.data import Data, HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform


@functional_transform('constant')
class Constant(BaseTransform):
    r"""Appends a constant value to each node feature :obj:`x`
    (functional name: :obj:`constant`).

    Args:
        value (float, optional): The value to add. (default: :obj:`1.0`)
        cat (bool, optional): If set to :obj:`False`, existing node features
            will be replaced. (default: :obj:`True`)
        node_types (str or List[str], optional): The specified node type(s) to
            append constant values for if used on heterogeneous graphs.
            If set to :obj:`None`, constants will be added to each node feature
            :obj:`x` for all existing node types. (default: :obj:`None`)
    """
    def __init__(
        self,
        value: float = 1.0,
        cat: bool = True,
        node_types: Optional[Union[str, List[str]]] = None,
    ):
        if isinstance(node_types, str):
            node_types = [node_types]

        self.value = value
        self.cat = cat
        self.node_types = node_types

    def forward(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:

        for store in data.node_stores:
            if self.node_types is None or store._key in self.node_types:
                num_nodes = store.num_nodes
                c = torch.full((num_nodes, 1), self.value, dtype=torch.float)

                if hasattr(store, 'x') and self.cat:
                    x = store.x.view(-1, 1) if store.x.dim() == 1 else store.x
                    store.x = torch.cat([x, c.to(x.device, x.dtype)], dim=-1)
                else:
                    store.x = c

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(value={self.value})'
