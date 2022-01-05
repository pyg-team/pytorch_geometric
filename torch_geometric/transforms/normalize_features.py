from typing import List, Union

from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform


class NormalizeFeatures(BaseTransform):
    r"""Row-normalizes the attributes given in :obj:`attrs` to sum-up to one.

    Args:
        attrs (List[str]): The names of attributes to normalize.
            (default: :obj:`["x"]`)
    """
    def __init__(self, attrs: List[str] = ["x"]):
        self.attrs = attrs

    def __call__(self, data: Union[Data, HeteroData]):
        for store in data.stores:
            for key, value in store.items(*self.attrs):
                store[key] = value / value.sum(dim=-1, keepdim=True).clamp_(1.)
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
