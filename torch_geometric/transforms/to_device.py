from typing import List, Optional, Union

from torch_geometric.data import Data, HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform


@functional_transform('to_device')
class ToDevice(BaseTransform):
    r"""Performs tensor device conversion, either for all attributes of the
    :obj:`~torch_geometric.data.Data` object or only the ones given by
    :obj:`attrs` (functional name: :obj:`to_device`).

    Args:
        device (torch.device): The destination device.
        attrs (List[str], optional): If given, will only perform tensor device
            conversion for the given attributes. (default: :obj:`None`)
        non_blocking (bool, optional): If set to :obj:`True` and tensor
            values are in pinned memory, the copy will be asynchronous with
            respect to the host. (default: :obj:`False`)
    """
    def __init__(
        self,
        device: Union[int, str],
        attrs: Optional[List[str]] = None,
        non_blocking: bool = False,
    ) -> None:
        self.device = device
        self.attrs = attrs or []
        self.non_blocking = non_blocking

    def forward(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        return data.to(self.device, *self.attrs,
                       non_blocking=self.non_blocking)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.device})'
