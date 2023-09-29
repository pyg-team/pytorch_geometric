from typing import Tuple, Union

from torch_geometric.data import Data, HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform


@functional_transform('center')
class Center(BaseTransform):
    r"""Centers node positions :obj:`data.pos` around the origin
    (functional name: :obj:`center`).

    Args:
        interval (Tuple[float, float], optional): A tuple representing the
            interval for the transformation. Defaults to (0.0, 1.0).
    """
    def __init__(
            self,
            interval: Tuple[float, float] = (0.0, 1.0),
    ):
        self.interval = interval

    def forward(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        # Pass the interval argument to the transformed data
        data.interval = self.interval

        for store in data.node_stores:
            if hasattr(store, 'pos'):
                store.pos = store.pos - store.pos.mean(dim=-2, keepdim=True)
        return data
