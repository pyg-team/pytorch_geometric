import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from torch_geometric.data import FeatureStore, TensorAttr
from torch_geometric.data.feature_store import _field_status


@dataclass
class LocalTensorAttr(TensorAttr):
    r"""Tensor attribute for storing features without :obj:`index`."""
    def __init__(
        self,
        group_name: Optional[str] = _field_status.UNSET,
        attr_name: Optional[str] = _field_status.UNSET,
        index=None,
    ):
        super().__init__(group_name, attr_name, index)


class LocalFeatureStore(FeatureStore):
    r"""This class implements the :class:`torch_geometric.data.FeatureStore`
    interface to act as a local feature store for distributed training."""
    def __init__(self):
        super().__init__(tensor_attr_cls=LocalTensorAttr)

        self._feat: Dict[Tuple[str, str], Tensor] = {}

        # Save the global node/edge IDs:
        self._global_id: Dict[Tuple[str, str], Tensor] = {}

        # Save the mapping from global node/edge IDs to indices in `_feat`:
        self._global_id_to_index: Dict[Tuple[str, str], Tensor] = {}

    @staticmethod
    def key(attr: TensorAttr) -> Tuple[str, str]:
        return (attr.group_name, attr.attr_name)

    def put_global_id(self, global_id: Tensor, *args, **kwargs) -> bool:
        attr = self._tensor_attr_cls.cast(*args, **kwargs)
        self._global_id[self.key(attr)] = global_id
        self._set_global_id_to_index(attr)
        return True

    def get_global_id(self, *args, **kwargs) -> Optional[Tensor]:
        attr = self._tensor_attr_cls.cast(*args, **kwargs)
        return self._global_id.get(self.key(attr))

    def remove_global_id(self, *args, **kwargs) -> bool:
        attr = self._tensor_attr_cls.cast(*args, **kwargs)
        return self._global_id.pop(self.key(attr), None) is not None

    def _set_global_id_to_index(self, *args, **kwargs):
        attr = self._tensor_attr_cls.cast(*args, **kwargs)
        global_id = self.get_global_id(attr)

        if global_id is None:
            return

        # TODO Compute this mapping without materializing a full-sized tensor:
        global_id_to_index = global_id.new_full((int(global_id.max()) + 1, ),
                                                fill_value=-1)
        global_id_to_index[global_id] = torch.arange(global_id.numel())
        self._global_id_to_index[self.key(attr)] = global_id_to_index

    def _put_tensor(self, tensor: Tensor, attr: TensorAttr) -> bool:
        assert attr.index is None
        self._feat[self.key(attr)] = tensor
        return True

    def _get_tensor(self, attr: TensorAttr) -> Optional[Tensor]:
        tensor = self._feat.get(self.key(attr))

        if tensor is None:
            return None

        if attr.index is None:  # Empty indices return the full tensor:
            return tensor

        return tensor[attr.index]

    def _remove_tensor(self, attr: TensorAttr) -> bool:
        assert attr.index is None
        return self._feat.pop(self.key(attr), None) is not None

    def get_tensor_from_global_id(self, *args, **kwargs) -> Optional[Tensor]:
        attr = self._tensor_attr_cls.cast(*args, **kwargs)
        assert attr.index is not None

        attr = copy.copy(attr)
        attr.index = self._global_id_to_index[self.key(attr)][attr.index]

        return self.get_tensor(attr)

    def _get_tensor_size(self, attr: TensorAttr) -> Tuple[int, ...]:
        return self._get_tensor(attr).size()

    def get_all_tensor_attrs(self) -> List[LocalTensorAttr]:
        return [self._tensor_attr_cls.cast(*key) for key in self._feat.keys()]
