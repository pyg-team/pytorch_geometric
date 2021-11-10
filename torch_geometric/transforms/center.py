from typing import Union

from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform


class Center(BaseTransform):
    r"""Centers node positions :obj:`pos` around the origin."""
    def __call__(self, data: Union[Data, HeteroData]):
        for store in data.node_stores:
            if hasattr(store, 'pos'):
                store.pos = store.pos - store.pos.mean(dim=-2, keepdim=True)
        return data
