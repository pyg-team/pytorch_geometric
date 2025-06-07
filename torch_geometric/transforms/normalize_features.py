from typing import List, Optional, Union

from torch_geometric.data import Data, HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform


@functional_transform('normalize_features')
class NormalizeFeatures(BaseTransform):
    r"""Row-normalizes the attributes given in :obj:`attrs` to sum-up to one
    (functional name: :obj:`normalize_features`).

    Args:
        attrs (List[str]): The names of attributes to normalize.
            (default: :obj:`["x"]`)
    """
    def __init__(self, attrs: Optional[List[str]] = None) -> None:
        self.attrs = attrs or ["x"]

    def forward(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        for store in data.stores:
            for key, value in store.items(*self.attrs):
                if value.numel() > 0:
                    value = value - value.min()
                    value.div_(value.sum(dim=-1, keepdim=True).clamp_(min=1.))
                    store[key] = value
        return data
