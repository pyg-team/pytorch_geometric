from typing import Optional, Union

import torch
import torch.distributed as dist
from nv_distributed_graph import (
    DistEmbedding,
    DistTensor,
    dist_shmem,
    nvlink_network,
)

import torch_geometric
from torch_geometric.data.feature_store import FeatureStore, TensorAttr


class WholeGraphFeatureStore(FeatureStore):
    r"""A high-performance, UVA-enabled, and multi-GPU/multi-node friendly feature store, powered by WholeGraph library.
    It is compatible with PyG's FeatureStore class and supports both homogeneous and heterogeneous graph data types.

    Args:
        pyg_data (torch_geometric.data.Data or torch_geometric.data.HeteroData): The input PyG graph data.

    Attributes:
        _store (dict): A dictionary to hold the feature embeddings.
        backend (str): Using 'nccl' or 'vmm' backend for inter-GPU communication if applicable.

    Methods:
        _put_tensor(tensor, attr):
            Puts a tensor into the feature store.
        _get_tensor(attr):
            Retrieves a tensor from the feature store with a given set of indexes.
        _remove_tensor(attr):
            Not yet implemented; intended for compatibility with PyG's FeatureStore class.
        _get_tensor_size(attr):
            Returns the size of a tensor in the feature store.
        get_all_tensor_attrs():
            Obtains all feature attributes stored in the feature store.
    """
    def __init__(self, pyg_data):
        r"""Initializes the WholeGraphFeatureStore class and loads features from torch_geometric.data.Data/HeteroData."""
        super().__init__()
        self._store = {
        }  # A dictionary of tuple to hold the feature embeddings

        if dist_shmem.get_local_size() == dist.get_world_size():
            self.backend = 'vmm'
        else:
            self.backend = 'vmm' if nvlink_network() else 'nccl'

        if isinstance(pyg_data, torch_geometric.data.Data):
            self.put_tensor(pyg_data['x'], group_name=None, attr_name='x',
                            index=None)
            self.put_tensor(pyg_data['y'], group_name=None, attr_name='y',
                            index=None)

        elif isinstance(pyg_data, torch_geometric.data.HeteroData
                        ):  # if HeteroData, we need to handle differently
            for group_name, group in pyg_data.node_items():
                for attr_name in group:
                    if group.is_node_attr(attr_name) and attr_name in {
                            'x', 'y'
                    }:
                        self.put_tensor(pyg_data[group_name][attr_name],
                                        group_name=group_name,
                                        attr_name=attr_name, index=None)
        else:
            raise TypeError(
                "Expected pyg_data to be of type torch_geometric.data.Data or torch_geometric.data.HeteroData."
            )

    def _put_tensor(self, tensor: torch.Tensor, attr):
        """Creates and stores features (either DistTensor or DistEmbedding) from the given tensor,
        using a key derived from the group and attribute name.
        """
        key = (attr.group_name, attr.attr_name)
        out = self._store.get(key)
        if out is not None and attr.index is not None:
            out[attr.index] = tensor
        else:
            assert attr.index is None
            if tensor.dim() == 1:
                # No need to unsqueeze if WholeGraph fix this https://github.com/rapidsai/wholegraph/pull/229
                self._store[key] = DistTensor(tensor.unsqueeze(1),
                                              device="cpu",
                                              backend=self.backend)
            else:
                self._store[key] = DistEmbedding(tensor, device="cpu",
                                                 backend=self.backend)
        return True

    def _get_tensor(
            self,
            attr) -> Optional[Union[torch.Tensor, DistTensor, DistEmbedding]]:
        """Retrieves a tensor based on the provided attribute.

        Args:
            attr: An object containing the necessary attributes to fetch the tensor.

        Returns:
            A tensor which can be of type torch.Tensor, DistTensor, or DistEmbedding, or None if not found.
        """
        key = (attr.group_name, attr.attr_name)
        tensor = self._store.get(key)
        if tensor is not None:
            if attr.index is not None:
                output = tensor[attr.index]
                return output
            else:
                return tensor
        return None

    def _remove_tensor(self, attr):
        pass

    def _get_tensor_size(self, attr):
        return self._get_tensor(attr).shape

    def get_all_tensor_attrs(self):
        r"""Obtains all feature attributes stored in `Data`."""
        return [
            TensorAttr(group_name=group, attr_name=name)
            for group, name in self._store.keys()
        ]
