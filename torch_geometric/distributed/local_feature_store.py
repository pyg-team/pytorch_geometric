import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from torch_geometric.data import FeatureStore, TensorAttr
from torch_geometric.data.feature_store import _field_status
from torch_geometric.typing import EdgeType, NodeType


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

    # Initialization ##########################################################

    @classmethod
    def from_homogeneous_graph(
        cls,
        node_id: Tensor,
        x: Optional[Tensor] = None,
        y: Optional[Tensor] = None,
        edge_attr: Optional[Tensor] = None,
    ) -> 'LocalFeatureStore':
        r"""Creates a local feature store from homogeneous :pyg:`PyG` tensors.

        Args:
            node_id (torch.Tensor): The global identifier for every local node.
            x (torch.Tensor, optional): The node features.
                (default: :obj:`None`)
            y (torch.Tensor, optional): The node labels. (default: :obj:`None`)
            edge_attr (torch.Tensor, optional): The edge features.
                (default: :obj:`None`)
        """
        feat_store = cls()
        feat_store.put_global_id(node_id, group_name=None)
        if x is not None:
            feat_store.put_tensor(x, group_name=None, attr_name='x')
        if y is not None:
            feat_store.put_tensor(y, group_name=None, attr_name='y')
        if edge_attr is not None:
            feat_store.put_tensor(edge_attr, group_name=None, attr_name='y')
        return feat_store

    @classmethod
    def from_heterogeneous_graph(
        cls,
        node_id_dict: Dict[NodeType, Tensor],
        x_dict: Optional[Dict[NodeType, Tensor]] = None,
        y_dict: Optional[Dict[NodeType, Tensor]] = None,
        edge_attr_dict: Optional[Dict[EdgeType, Tensor]] = None,
    ) -> 'LocalFeatureStore':
        r"""Creates a local graph store from heterogeneous :pyg:`PyG` tensors.

        Args:
            node_id_dict (Dict[NodeType, torch.Tensor]): The global identifier
                for every local node of every node type.
            x_dict (Dict[NodeType, torch.Tensor], optional): The node features
                of node types. (default: :obj:`None`)
            y_dict (Dict[NodeType, torch.Tensor], optional): The node labels of
                node types. (default: :obj:`None`)
            edge_attr_dict (Dict[EdgeType, torch.Tensor], optional): The edge
                features of edge types. (default: :obj:`None`)
        """
        feat_store = cls()

        for node_type, node_id in node_id_dict.items():
            feat_store.put_global_id(node_id, group_name=node_type)
        if x_dict is not None:
            for node_type, x in x_dict.items():
                feat_store.put_tensor(x, group_name=node_type, attr_name='x')
        if y_dict is not None:
            for node_type, y in y_dict.items():
                feat_store.put_tensor(y, group_name=node_type, attr_name='y')
        if edge_attr_dict is not None:
            for edge_type, edge_attr in edge_attr_dict.items():
                feat_store.put_tensor(edge_attr, group_name=edge_type,
                                      attr_name='edge_attr')

        return feat_store
