from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from torch_geometric.data import FeatureStore, TensorAttr
from torch_geometric.typing import FeatureTensorType


class LocalFeatureStore(FeatureStore):
    r"""This class implements the :class:`torch_geometric.data.FeatureStore`
    interface to act as a local feature store for distributed training.
    """
    def __init__(self):

        self.store: Dict[Tuple[str, str], Tensor] = {}

        # save the global node/edge ids
        self.global_idx: Dict[Tuple[str, str], Tensor] = {}

        # save the the mapping from global sampled node/edge ids to local feature index
        self.id2index: Dict[Tuple[str, str], Tensor] = {}

        super().__init__()

    @staticmethod
    def key(attr: TensorAttr) -> str:
        return (attr.group_name, attr.attr_name)

    def set_global_ids(self, global_ids: Tensor, group_name: str,
                       attr_name: str):
        key_name = (group_name, attr_name)
        self.global_idx[key_name] = global_ids

    def get_global_ids(self, group_name: str, attr_name: str) -> Tensor:
        key_name = (group_name, attr_name)
        return self.global_idx.get(key_name)

    def set_id2index(self, group_name: str, attr_name: str):
        key_name = (group_name, attr_name)

        global_ids = self.global_idx.get(key_name)
        max_id = torch.max(global_ids).item()
        id2idx = torch.zeros(max_id + 1, dtype=torch.int64)
        id2idx[global_ids] = torch.arange(global_ids.size(0),
                                          dtype=torch.int64)

        self.id2index[key_name] = id2idx

    def get_id2index(self, group_name: str, attr_name: str) -> Tensor:
        key_name = (group_name, attr_name)
        return self.id2index.get(key_name)

    def set_global_ids_plus_id2index(self, global_ids: Tensor, group_name: str,
                                     attr_name: str):
        key_name = (group_name, attr_name)
        self.global_idx[key_name] = global_ids

        max_id = torch.max(global_ids).item()
        id2idx = torch.zeros(max_id + 1, dtype=torch.int64)
        id2idx[global_ids] = torch.arange(global_ids.size(0),
                                          dtype=torch.int64)

        self.id2index[key_name] = id2idx

    def _put_tensor(self, tensor: FeatureTensorType, attr: TensorAttr) -> bool:
        index = attr.index

        # None indices define the obvious index:
        if index is None:
            index = torch.arange(0, tensor.shape[0])

        # Store the index:
        self.store[LocalFeatureStore.key(attr)] = (index, tensor)

        return True

    def _get_tensor(self, attr: TensorAttr) -> Optional[FeatureTensorType]:
        index, tensor = self.store.get(LocalFeatureStore.key(attr),
                                       (None, None))
        if tensor is None:
            return None

        # None indices return the whole tensor:
        if attr.index is None:
            return tensor

        # Empty slices return the whole tensor:
        if (isinstance(attr.index, slice)
                and attr.index == slice(None, None, None)):
            return tensor

        #idx = (torch.cat([(index == v).nonzero() for v in attr.index]).view(-1)
        #       if attr.index.numel() > 0 else [])
        return tensor[attr.index]

    def _remove_tensor(self, attr: TensorAttr) -> bool:
        del self.store[LocalFeatureStore.key(attr)]
        return True

    def _get_tensor_size(self, attr: TensorAttr) -> Tuple:
        return self._get_tensor(attr).size()

    def get_all_tensor_attrs(self) -> List[str]:
        return [TensorAttr(*key) for key in self.store.keys()]

    def __len__(self):
        raise NotImplementedError
