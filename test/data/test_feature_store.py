from dataclasses import dataclass
from typing import Optional

import torch

from torch_geometric.data.feature_store import (
    AttrView,
    FeatureStore,
    TensorAttr,
    _field_status,
)
from torch_geometric.typing import FeatureTensorType


class MyFeatureStore(FeatureStore):
    def __init__(self):
        super().__init__()
        self.store = {}

    @classmethod
    def key(cls, attr: TensorAttr):
        return (attr.group_name or '', attr.attr_name or '')

    def _put_tensor(self, tensor: FeatureTensorType, attr: TensorAttr) -> bool:
        index = attr.index

        # Not set or None indices define the obvious index
        if not attr.is_set('index') or index is None:
            index = torch.arange(0, tensor.shape[0])

        # Store the index as a column
        self.store[MyFeatureStore.key(attr)] = torch.cat(
            (index.reshape(-1, 1), tensor), dim=1)

        return True

    def _get_tensor(self, attr: TensorAttr) -> Optional[FeatureTensorType]:
        tensor = self.store.get(MyFeatureStore.key(attr), None)
        if tensor is None:
            return None

        # Not set or None indices return the whole tensor
        if not attr.is_set('index') or attr.index is None:
            return tensor[:, 1:]

        # Index into the tensor
        indices = torch.cat([(tensor[:, 0] == v).nonzero()
                             for v in attr.index]).reshape(1, -1)[0]

        return torch.index_select(tensor[:, 1:], 0, indices)

    def _remove_tensor(self, attr: TensorAttr) -> bool:
        del self.store[MyFeatureStore.key(attr)]

    def __len__(self):
        raise NotImplementedError


@dataclass
class MyTensorAttrNoGroupName(TensorAttr):
    def __init__(self, attr_name=_field_status.UNSET,
                 index=_field_status.UNSET):
        # Treat group_name as optional, and move it to the end
        super().__init__(None, attr_name, index)


class MyFeatureStoreNoGroupName(MyFeatureStore):
    def __init__(self):
        super().__init__()
        self._attr_cls = MyTensorAttrNoGroupName

    @classmethod
    def key(cls, attr: TensorAttr):
        return attr.attr_name or ''

    def __len__(self):
        raise NotImplementedError


def test_feature_store():
    r"""Tests basic API and indexing functionality of a feature store."""
    store = MyFeatureStore()
    tensor = torch.Tensor([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    index = torch.Tensor([0, 1, 2])

    attr_name = 'feat'
    group_name = 'A'

    attr = TensorAttr(group_name, attr_name, index)

    # Normal API
    store.put_tensor(tensor, attr)
    assert torch.equal(store.get_tensor(attr), tensor)
    assert torch.equal(
        store.get_tensor((group_name, attr_name, torch.Tensor([0, 2]))),
        tensor[[0, 2]],
    )
    assert store.get_tensor(TensorAttr(index=index)) is None
    store.remove_tensor(TensorAttr(group_name, attr_name))
    assert store.get_tensor(attr) is None

    # Views
    view = store.view(TensorAttr(group_name=group_name))
    view.attr_name = attr_name
    view.index = index
    assert view == AttrView(store, TensorAttr(group_name, attr_name, index))

    # Indexing

    # Setting via indexing
    store[group_name, attr_name, index] = tensor

    # Fully-specified forms, all of which produce a tensor output
    assert torch.equal(store[group_name, attr_name, index], tensor)
    assert torch.equal(store[group_name, attr_name, None], tensor)
    assert torch.equal(store[group_name, attr_name, :], tensor)
    assert torch.equal(store[group_name].feat[:], tensor)

    # Partially-specified forms, which produce an AttrView object
    assert store[group_name] == store.view(TensorAttr(group_name=group_name))
    assert store[group_name].feat == store.view(
        TensorAttr(group_name=group_name, attr_name=attr_name))

    # Partially-specified forms, when called, produce a Tensor output
    # from the `TensorAttr` that has been partially specified.
    store[group_name] = tensor
    assert isinstance(store[group_name], AttrView)
    assert torch.equal(store[group_name](), tensor)

    # Deletion
    del store[group_name, attr_name, index]
    assert store[group_name, attr_name, index] is None
    del store[group_name]
    assert store[group_name]() is None


def test_feature_store_override():
    store = MyFeatureStoreNoGroupName()
    tensor = torch.Tensor([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    index = torch.Tensor([0, 1, 2])

    attr_name = 'feat'

    # Only use attr_name and index, in that order
    store[attr_name, index] = tensor

    # A few assertions to ensure group_name is not needed
    assert isinstance(store[attr_name], AttrView)
    assert torch.equal(store[attr_name, index], tensor)
    assert torch.equal(store[attr_name][index], tensor)
    assert torch.equal(store[attr_name][:], tensor)
    assert torch.equal(store[attr_name, :], tensor)
