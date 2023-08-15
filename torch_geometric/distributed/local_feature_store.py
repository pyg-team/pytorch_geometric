import copy
import json
import os.path as osp
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

from torch_geometric.data import FeatureStore, TensorAttr
from torch_geometric.data.feature_store import _FieldStatus
from torch_geometric.distributed.rpc import (
    RPCCallBase,
    RPCRouter,
    rpc_async,
    rpc_register,
)
from torch_geometric.typing import EdgeType, NodeOrEdgeType, NodeType


class RPCCallFeatureLookup(RPCCallBase):
    r"""A wrapper for RPC calls to the feature store."""
    def __init__(self, dist_feature: FeatureStore):
        super().__init__()
        self.dist_feature = dist_feature

    def rpc_async(self, *args, **kwargs):
        return self.dist_feature.rpc_local_feature_get(*args, **kwargs)

    def rpc_sync(self, *args, **kwargs):
        raise NotImplementedError


@dataclass
class LocalTensorAttr(TensorAttr):
    r"""Tensor attribute for storing features without :obj:`index`."""
    def __init__(
        self,
        group_name: Optional[Union[NodeType, EdgeType]] = _FieldStatus.UNSET,
        attr_name: Optional[str] = _FieldStatus.UNSET,
        index=None,
    ):
        super().__init__(group_name, attr_name, index)


class LocalFeatureStore(FeatureStore):
    r"""This class implements the :class:`torch_geometric.data.FeatureStore`
    interface to act as a local feature store for distributed training."""
    def __init__(self):
        super().__init__(tensor_attr_cls=LocalTensorAttr)

        self._feat: Dict[Tuple[Union[NodeType, EdgeType], str], Tensor] = {}

        # Save the global node/edge IDs:
        self._global_id: Dict[Union[NodeType, EdgeType], Tensor] = {}

        # Save the mapping from global node/edge IDs to indices in `_feat`:
        self._global_id_to_index: Dict[Union[NodeType, EdgeType], Tensor] = {}

        # For partition/rpc information related to distribute features:
        self.num_partitions = 1
        self.partition_idx = 0
        self.feature_pb: Union[Tensor, Dict[NodeOrEdgeType, Tensor]]
        self.local_only = False
        self.rpc_router: Optional[RPCRouter] = None
        self.meta: Optional[Dict] = None
        self.rpc_call_id: Optional[int] = None

    @staticmethod
    def key(attr: TensorAttr) -> Tuple[str, str]:
        return (attr.group_name, attr.attr_name)

    def put_global_id(
        self,
        global_id: Tensor,
        group_name: Union[NodeType, EdgeType],
    ) -> bool:
        self._global_id[group_name] = global_id
        self._set_global_id_to_index(group_name)
        return True

    def get_global_id(
        self,
        group_name: Union[NodeType, EdgeType],
    ) -> Optional[Tensor]:
        return self._global_id.get(group_name)

    def remove_global_id(self, group_name: Union[NodeType, EdgeType]) -> bool:
        return self._global_id.pop(group_name) is not None

    def _set_global_id_to_index(self, group_name: Union[NodeType, EdgeType]):
        global_id = self.get_global_id(group_name)

        if global_id is None:
            return

        # TODO Compute this mapping without materializing a full-sized tensor:
        global_id_to_index = global_id.new_full((int(global_id.max()) + 1, ),
                                                fill_value=-1)
        global_id_to_index[global_id] = torch.arange(global_id.numel())
        self._global_id_to_index[group_name] = global_id_to_index

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
        attr.index = self._global_id_to_index[attr.group_name][attr.index]

        return self.get_tensor(attr)

    def _get_tensor_size(self, attr: TensorAttr) -> Tuple[int, ...]:
        return self._get_tensor(attr).size()

    def get_all_tensor_attrs(self) -> List[LocalTensorAttr]:
        return [self._tensor_attr_cls.cast(*key) for key in self._feat.keys()]

    def set_rpc_router(self, rpc_router: RPCRouter):
        self.rpc_router = rpc_router

        if not self.local_only:
            if self.rpc_router is None:
                raise ValueError("An RPC router must be provided")
            rpc_call = RPCCallFeatureLookup(self)
            self.rpc_call_id = rpc_register(rpc_call)
        else:
            self.rpc_call_id = None

    def lookup_features(
        self,
        index: Tensor,
        is_node_feat: bool = True,
        input_type: Optional[NodeOrEdgeType] = None,
    ) -> torch.futures.Future:
        r"""Lookup of local/remote features."""
        remote_fut = self._remote_lookup_features(index, is_node_feat,
                                                  input_type)
        local_feature = self._local_lookup_features(index, is_node_feat,
                                                    input_type)
        res_fut = torch.futures.Future()

        def when_finish(*_):
            try:
                remote_feature_list = remote_fut.wait()
                # combine the feature from remote and local
                result = torch.zeros(index.size(0), local_feature[0].size(1),
                                     dtype=local_feature[0].dtype)
                result[local_feature[1]] = local_feature[0]
                for remote in remote_feature_list:
                    result[remote[1]] = remote[0]
            except Exception as e:
                res_fut.set_exception(e)
            else:
                res_fut.set_result(result)

        remote_fut.add_done_callback(when_finish)
        return res_fut

    def _local_lookup_features(
        self,
        index: Tensor,
        is_node_feat: bool = True,
        input_type: Optional[Union[NodeType, EdgeType]] = None,
    ) -> Tuple[Tensor, Tensor]:
        r""" lookup the features in local nodes based on node/edge ids """
        if self.meta['is_hetero']:
            feat = self
            pb = self.feature_pb[input_type]
        else:
            feat = self
            pb = self.feature_pb

        input_order = torch.arange(index.size(0), dtype=torch.long)
        partition_ids = pb[index]

        local_mask = partition_ids == self.partition_idx
        local_ids = torch.masked_select(index, local_mask)
        local_index = torch.masked_select(input_order, local_mask)

        if self.meta["is_hetero"]:
            if is_node_feat:
                kwargs = dict(group_name=input_type, attr_name='x')
                ret_feat = feat.get_tensor_from_global_id(
                    index=local_ids, **kwargs)
            else:
                kwargs = dict(group_name=input_type, attr_name='edge_attr')
                ret_feat = feat.get_tensor_from_global_id(
                    index=local_ids, **kwargs)
        else:
            if is_node_feat:
                kwargs = dict(group_name=None, attr_name='x')
                ret_feat = feat.get_tensor_from_global_id(
                    index=local_ids, **kwargs)
            else:
                kwargs = dict(group_name=(None, None), attr_name='edge_attr')
                ret_feat = feat.get_tensor_from_global_id(
                    index=local_ids, **kwargs)

        return ret_feat, local_index

    def _remote_lookup_features(
        self,
        index: Tensor,
        is_node_feat: bool = True,
        input_type: Optional[Union[NodeType, EdgeType]] = None,
    ) -> torch.futures.Future:
        r"""Fetch the remote features with the remote node/edge ids"""

        if self.meta["is_hetero"]:
            pb = self.feature_pb[input_type]
        else:
            pb = self.feature_pb

        input_order = torch.arange(index.size(0), dtype=torch.long)
        partition_ids = pb[index]
        futs, indexes = [], []
        for pidx in range(0, self.num_partitions):
            if pidx == self.partition_idx:
                continue
            remote_mask = (partition_ids == pidx)
            remote_ids = index[remote_mask]
            if remote_ids.shape[0] > 0:
                to_worker = self.rpc_router.get_to_worker(pidx)
                futs.append(
                    rpc_async(
                        to_worker,
                        self.rpc_call_id,
                        args=(remote_ids.cpu(), is_node_feat, input_type),
                    ))
                indexes.append(torch.masked_select(input_order, remote_mask))
        collect_fut = torch.futures.collect_all(futs)
        res_fut = torch.futures.Future()

        def when_finish(*_):
            try:
                fut_list = collect_fut.wait()
                result = []
                for i, fut in enumerate(fut_list):
                    result.append((fut.wait(), indexes[i]))
            except Exception as e:
                res_fut.set_exception(e)
            else:
                res_fut.set_result(result)

        collect_fut.add_done_callback(when_finish)
        return res_fut

    def rpc_local_feature_get(
        self,
        index: Tensor,
        is_node_feat: bool = True,
        input_type: Optional[Union[NodeType, EdgeType]] = None,
    ) -> Tensor:
        r"""Lookup of features in remote nodes."""
        if self.meta['is_hetero']:
            feat = self
            if is_node_feat:
                kwargs = dict(group_name=input_type, attr_name='x')
                ret_feat = feat.get_tensor_from_global_id(
                    index=index, **kwargs)
            else:
                kwargs = dict(group_name=input_type, attr_name='edge_attr')
                ret_feat = feat.get_tensor_from_global_id(
                    index=index, **kwargs)
        else:
            feat = self
            if is_node_feat:
                kwargs = dict(group_name=None, attr_name='x')
                ret_feat = feat.get_tensor_from_global_id(
                    index=index, **kwargs)
            else:
                kwargs = dict(group_name=(None, None), attr_name='edge_attr')
                ret_feat = feat.get_tensor_from_global_id(
                    index=index, **kwargs)

        return ret_feat

    # Initialization ##########################################################

    @classmethod
    def from_data(
        cls,
        node_id: Tensor,
        x: Optional[Tensor] = None,
        y: Optional[Tensor] = None,
        edge_id: Optional[Tensor] = None,
        edge_attr: Optional[Tensor] = None,
    ) -> 'LocalFeatureStore':
        r"""Creates a local feature store from homogeneous :pyg:`PyG` tensors.

        Args:
            node_id (torch.Tensor): The global identifier for every local node.
            x (torch.Tensor, optional): The node features.
                (default: :obj:`None`)
            y (torch.Tensor, optional): The node labels. (default: :obj:`None`)
            edge_id (torch.Tensor, optional): The global identifier for every
                local edge. (default: :obj:`None`)
            edge_attr (torch.Tensor, optional): The edge features.
                (default: :obj:`None`)
        """
        feat_store = cls()
        feat_store.put_global_id(node_id, group_name=None)
        if x is not None:
            feat_store.put_tensor(x, group_name=None, attr_name='x')
        if y is not None:
            feat_store.put_tensor(y, group_name=None, attr_name='y')
        if edge_id is not None:
            feat_store.put_global_id(edge_id, group_name=(None, None))
        if edge_attr is not None:
            if edge_id is None:
                raise ValueError("'edge_id' needs to be present in case "
                                 "'edge_attr' is passed")
            feat_store.put_tensor(edge_attr, group_name=(None, None),
                                  attr_name='edge_attr')
        return feat_store

    @classmethod
    def from_hetero_data(
        cls,
        node_id_dict: Dict[NodeType, Tensor],
        x_dict: Optional[Dict[NodeType, Tensor]] = None,
        y_dict: Optional[Dict[NodeType, Tensor]] = None,
        edge_id_dict: Optional[Dict[EdgeType, Tensor]] = None,
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
            edge_id_dict (Dict[EdgeType, torch.Tensor], optional): The global
                identifier for every local edge of edge types.
                (default: :obj:`None`)
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
        if edge_id_dict is not None:
            for edge_type, edge_id in edge_id_dict.items():
                feat_store.put_global_id(edge_id, group_name=edge_type)
        if edge_attr_dict is not None:
            for edge_type, edge_attr in edge_attr_dict.items():
                if edge_id_dict is None or edge_type not in edge_id_dict:
                    raise ValueError("'edge_id' needs to be present in case "
                                     "'edge_attr' is passed")
                feat_store.put_tensor(edge_attr, group_name=edge_type,
                                      attr_name='edge_attr')

        return feat_store

    @classmethod
    def from_partition(cls, root: str, pid: int) -> 'LocalFeatureStore':
        with open(osp.join(root, 'META.json'), 'r') as f:
            meta = json.load(f)

        part_dir = osp.join(root, f'part_{pid}')
        assert osp.exists(part_dir)

        node_feats: Optional[Dict[str, Any]] = None
        if osp.exists(osp.join(part_dir, 'node_feats.pt')):
            node_feats = torch.load(osp.join(part_dir, 'node_feats.pt'))

        edge_feats: Optional[Dict[str, Any]] = None
        if osp.exists(osp.join(part_dir, 'edge_feats.pt')):
            edge_feats = torch.load(osp.join(part_dir, 'edge_feats.pt'))

        feat_store = cls()

        if not meta['is_hetero'] and node_feats is not None:
            feat_store.put_global_id(node_feats['global_id'], group_name=None)
            for key, value in node_feats['feats'].items():
                feat_store.put_tensor(value, group_name=None, attr_name=key)

        if not meta['is_hetero'] and edge_feats is not None:
            feat_store.put_global_id(edge_feats['global_id'],
                                     group_name=(None, None))
            for key, value in edge_feats['feats'].items():
                feat_store.put_tensor(value, group_name=(None, None),
                                      attr_name=key)

        if meta['is_hetero'] and node_feats is not None:
            for node_type, node_feat in node_feats.items():
                feat_store.put_global_id(node_feat['global_id'],
                                         group_name=node_type)
                for key, value in node_feat['feats'].items():
                    feat_store.put_tensor(value, group_name=node_type,
                                          attr_name=key)

        if meta['is_hetero'] and edge_feats is not None:
            for edge_type, edge_feat in edge_feats.items():
                feat_store.put_global_id(edge_feat['global_id'],
                                         group_name=edge_type)
                for key, value in edge_feat['feats'].items():
                    feat_store.put_tensor(value, group_name=edge_type,
                                          attr_name=key)

        return feat_store
