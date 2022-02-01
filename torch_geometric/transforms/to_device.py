from typing import List, Optional, Union

from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform


class ToDevice(BaseTransform):
    r"""Performs tensor device conversion, either for all attributes of the
    :obj:`~torch_geometric.data.Data` object or only the ones given by
    :obj:`attrs`.

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
    ):
        self.device = device
        self.attrs = attrs or []
        self.non_blocking = non_blocking

    def __call__(self, data: Union[Data, HeteroData]):
        return data.to(self.device, *self.attrs,
                       non_blocking=self.non_blocking)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.device})'
