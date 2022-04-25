from typing import Optional

import torch

from torch_geometric.data.feature_store import FeatureStore, TensorAttr
from torch_geometric.typing import FeatureTensorType


class MyFeatureStore(FeatureStore):
    r"""A basic feature store, does NOT implement all functionality of a
    fully-fledged feature store. Only works for Torch tensors."""
    def __init__(self):
        super().__init__(backend='test')
        self.store = {}

    @classmethod
    def key(cls, attr: TensorAttr):
        return (attr.tensor_type or '', attr.node_type or '', attr.graph_type
                or '')

    def _put_tensor(self, tensor: FeatureTensorType, attr: TensorAttr) -> bool:
        self.store[MyFeatureStore.key(attr)] = torch.cat(
            (attr.index.reshape(-1, 1), tensor), dim=1)
        return True

    def _get_tensor(self, attr: TensorAttr) -> Optional[FeatureTensorType]:
        tensor = self.store.get(MyFeatureStore.key(attr), None)
        if tensor is None:
            return None
        if attr.index is not None:
            indices = torch.cat([(tensor[:, 0] == v).nonzero()
                                 for v in attr.index]).reshape(1, -1)[0]

            return torch.index_select(tensor[:, 1:], 0, indices)
        return tensor[:, 1:]

    def _remove_tensor(self, attr: TensorAttr) -> bool:
        if attr.index is not None:
            raise NotImplementedError
        del self.store[MyFeatureStore.key(attr)]

    def __len__(self):
        raise NotImplementedError


def test_feature_store():
    r"""Tests basic API and indexing functionality of a feature store."""

    store = MyFeatureStore()
    tensor = torch.Tensor([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    index = torch.Tensor([0, 1, 2])

    tensor_type = 'feat'
    node_type = 'A'
    graph_type = 'graph'

    attr = TensorAttr(index, tensor_type, node_type, graph_type)

    # Normal API
    store.put_tensor(tensor, attr)
    assert torch.equal(store.get_tensor(attr), tensor)
    assert torch.equal(
        store.get_tensor(
            (torch.Tensor([0, 2]), tensor_type, node_type, graph_type)),
        tensor[[0, 2]],
    )
    assert store.get_tensor((index)) is None
    store.remove_tensor((None, tensor_type, node_type, graph_type))
    assert store.get_tensor(attr) is None

    # Indexing
    store[attr] = tensor
    assert torch.equal(store[attr], tensor)
    assert torch.equal(
        store[(torch.Tensor([0, 2]), tensor_type, node_type, graph_type)],
        tensor[[0, 2]],
    )
    assert store[(index)] is None
    del store[(None, tensor_type, node_type, graph_type)]
    assert store.get_tensor(attr) is None

    # Advanced indexing
    store[attr] = tensor
    assert (torch.equal(
        store[TensorAttr(node_type=node_type,
                         graph_type=graph_type)].feat[index], tensor))
