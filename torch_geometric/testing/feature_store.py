from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from torch_geometric.data import FeatureStore, TensorAttr
from torch_geometric.typing import FeatureTensorType


class MyFeatureStore(FeatureStore):
    def __init__(self):
        super().__init__()
        self.store: Dict[Tuple[str, str], Tensor] = {}

    @staticmethod
    def key(attr: TensorAttr) -> str:
        return (attr.group_name, attr.attr_name)

    def _put_tensor(self, tensor: FeatureTensorType, attr: TensorAttr) -> bool:
        index = attr.index

        # None indices define the obvious index:
        if index is None:
            index = torch.arange(0, tensor.shape[0])

        # Store the index:
        self.store[self.key(attr)] = (index, tensor)

        return True

    def _get_tensor(self, attr: TensorAttr) -> Optional[FeatureTensorType]:
        index, tensor = self.store.get(self.key(attr), (None, None))
        if tensor is None:
            return None

        # None indices return the whole tensor:
        if attr.index is None:
            return tensor

        # Empty slices return the whole tensor:
        if (isinstance(attr.index, slice)
                and attr.index == slice(None, None, None)):
            return tensor

        idx = (torch.cat([(index == v).nonzero() for v in attr.index]).view(-1)
               if attr.index.numel() > 0 else [])
        return tensor[idx]

    def _remove_tensor(self, attr: TensorAttr) -> bool:
        return self.store.pop(self.key(attr), None) is not None

    def _get_tensor_size(self, attr: TensorAttr) -> Tuple[int, ...]:
        return self._get_tensor(attr).size()

    def get_all_tensor_attrs(self) -> List[TensorAttr]:
        return [self._tensor_attr_cls.cast(*key) for key in self.store.keys()]

    def __len__(self):
        raise NotImplementedError
