from typing import Optional, Any, Tuple, List
import grpc
from client_server_proto import service_pb2, service_pb2_grpc
import numpy
import torch
from torch_geometric.data.feature_store import FeatureStore, TensorAttr
import os

from torch_geometric.typing import FeatureTensorType


class RocksDBFeatureStore(FeatureStore):
    r"""A feature store implementation backed by RocksDB. This implementation
    assumes that the gRPC service is already running at port localhost:50051.
    We provide methods to load parquet files to the feature store.

    Args:
        server_address: The address of the server binary, by default
            `localhost:50051`.
    """
    def __init__(
        self,
        server_address: str,
        tensor_attr_cls: Optional[Any] = None,
    ):
        super().__init__(tensor_attr_cls)

        self._server_address = server_address
        self._stub: service_pb2_grpc.FeatureStoreStub

        # gRPC stub channel is not fork-safe:
        self._initialize_stub()
        os.register_at_fork(before=lambda: setattr(self, '_stub', None))
        os.register_at_fork(after_in_child=lambda: self._initialize_stub())
        os.register_at_fork(after_in_parent=lambda: self._initialize_stub())

    def _initialize_stub(self) -> None:
        r"""Initializes the class stub with an open channel to the Server."""
        channel = grpc.insecure_channel(self._server_address)
        self._stub = service_pb2_grpc.FeatureStoreStub(channel)

    def load_parquet(self, node_type: str, path: str) -> None:
        r"""Loads a parquet file with 0..n-1 index to the feature store,
        with group name `node_type`. Note that the same node type cannot
        be loaded multiple times.

        Args:
            node_type: The group name of the feature to load into the
                feature store.
            path: The path of the parquet file on local disk.
        """
        load_req = service_pb2.LoadFeaturesRequest()
        load_req.type = node_type
        load_req.path = path

        # TODO(manan): handle errors more gracefully. For now, every
        # error will be raised:
        self._stub.LoadFeatures(load_req)

    def _put_tensor(self, tensor: FeatureTensorType, attr: TensorAttr) -> bool:
        r"""Putting individual tensors is not supported on this feature store."""
        raise NotImplementedError

    def _remove_tensor(self, attr: TensorAttr) -> bool:
        r"""Removing individual tensors after loading is not supported on this
        feature store."""
        raise NotImplementedError

    def _get_tensor_size(self, attr: TensorAttr) -> Optional[Tuple[int, ...]]:
        # TODO(manan): implement...
        raise NotImplementedError

    def get_all_tensor_attrs(self) -> List[TensorAttr]:
        # TODO(manan): implement...
        raise NotImplementedError

    def _get_tensor(self, attr: TensorAttr) -> Optional[FeatureTensorType]:
        r"""Fetches a tensor from the server."""
        keys_req = service_pb2.KeysRequest()
        # TODO support attr_name
        keys_req.type = attr.group_name
        if isinstance(attr.index, torch.Tensor):
            keys_req.idx.extend(attr.index.to(torch.int64).tolist())
        elif isinstance(attr.index, numpy.ndarray):
            keys_req.idx.extend(attr.index.astype('int64').tolist())
        elif isinstance(attr.index, slice):
            # TODO support with .indices()
            raise NotImplementedError
        elif isinstance(attr.index, int):
            keys_req.idx.extend([attr.index])
        else:
            raise NotImplementedError
        out: service_pb2.KeysResponse = self._stub.GetFeatures(keys_req)

        # TODO(manan): get an output as a Tensor, return to the user; can
        # safely assume no strings for now, though we can support and fix
        # later:
        print(out)
        return None
