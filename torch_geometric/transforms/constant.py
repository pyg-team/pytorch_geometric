from typing import Union

import torch

from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform


class Constant(BaseTransform):
    r"""Adds a constant value to each node feature :obj:`x`.

    Args:
        value (float, optional): The value to add. (default: :obj:`1.0`)
        cat (bool, optional): If set to :obj:`False`, all existing node
            features will be replaced. (default: :obj:`True`)
    """
    def __init__(self, value: float = 1.0, cat: bool = True):
        self.value = value
        self.cat = cat

    def __call__(self, data: Union[Data, HeteroData]):
        for store in data.node_stores:
            c = torch.full((store.num_nodes, 1), self.value, dtype=torch.float)

            if hasattr(store, 'x') and self.cat:
                x = store.x.view(-1, 1) if store.x.dim() == 1 else store.x
                store.x = torch.cat([x, c.to(x.device, x.dtype)], dim=-1)
            else:
                store.x = c

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(value={self.value})'
