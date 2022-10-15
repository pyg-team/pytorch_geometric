from typing import List, Optional, Union

import torch

from torch_geometric.data import Data, HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform


@functional_transform('constant')
class Constant(BaseTransform):
    r"""Adds a constant value to each node feature :obj:`x`
    (functional name: :obj:`constant`).

    Args:
        value (float, optional): The value to add. (default: :obj:`1.0`)
        cat (bool, optional): If set to :obj:`False`, the node
            features will be replaced. (default: :obj:`True`)
        node_types (str or List[str], optional): The specified node types to
            add constants, which is used only for heterogeneous graphs.
            If set to :obj:`None`, constants will be added to each node feature
            :obj:`x` of all existing node types. (default: :obj:`None`)
    """
    def __init__(self, value: float = 1.0, cat: bool = True,
                 node_types: Optional[Union[str, List[str]]] = None):
        self.value = value
        self.cat = cat
        self.node_types = node_types

    def __call__(self, data: Union[Data,
                                   HeteroData]) -> Union[Data, HeteroData]:
        if isinstance(data, HeteroData) and self.node_types is not None:
            node_types = [self.node_types] if isinstance(
                self.node_types, str) else self.node_types
            stores = [data[node_type] for node_type in node_types]
        else:
            stores = data.node_stores
        for store in stores:
            c = torch.full((store.num_nodes, 1), self.value, dtype=torch.float)

            if hasattr(store, 'x') and self.cat:
                x = store.x.view(-1, 1) if store.x.dim() == 1 else store.x
                store.x = torch.cat([x, c.to(x.device, x.dtype)], dim=-1)
            else:
                store.x = c

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(value={self.value})'
