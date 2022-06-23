from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pytest
import torch
from torch import Tensor

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
        self.store[MyFeatureStore.key(attr)] = (index, tensor)

        return True

    def _get_tensor(self, attr: TensorAttr) -> Optional[FeatureTensorType]:
        index, tensor = self.store.get(MyFeatureStore.key(attr), (None, None))
        if tensor is None:
            return None

        # None indices return the whole tensor:
        if attr.index is None:
            return tensor

        idx = torch.cat([(index == v).nonzero() for v in attr.index]).view(-1)
        return tensor[idx]

    def _remove_tensor(self, attr: TensorAttr) -> bool:
        del self.store[MyFeatureStore.key(attr)]
        return True

    def _get_tensor_size(self, attr: TensorAttr) -> Tuple:
        return self._get_tensor(attr).size()

    def get_all_tensor_attrs(self) -> List[str]:
        return [TensorAttr(*key) for key in self.store.keys()]

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
        self._tensor_attr_cls = MyTensorAttrNoGroupName


def test_feature_store():
    store = MyFeatureStore()
    tensor = torch.Tensor([[0, 0, 0], [1, 1, 1], [2, 2, 2]])

    group_name = 'A'
    attr_name = 'feat'
    index = torch.tensor([0, 1, 2])
    attr = TensorAttr(group_name, attr_name, index)

    # Normal API:
    store.put_tensor(tensor, attr)
    assert torch.equal(store.get_tensor(attr), tensor)
    assert torch.equal(
        store.get_tensor(group_name, attr_name, index=torch.tensor([0, 2])),
        tensor[torch.tensor([0, 2])],
    )
    store.remove_tensor(group_name, attr_name, None)
    with pytest.raises(KeyError):
        _ = store.get_tensor(attr)

    # Views:
    view = store.view(group_name=group_name)
    view.attr_name = attr_name
    view['index'] = index
    assert view == AttrView(store, TensorAttr(group_name, attr_name, index))

    # Indexing:
    store[group_name, attr_name, index] = tensor

    # Fully-specified forms, all of which produce a tensor output
    assert torch.equal(store[group_name, attr_name, index], tensor)
    assert torch.equal(store[group_name, attr_name, None], tensor)
    assert torch.equal(store[group_name, attr_name, :], tensor)
    assert torch.equal(store[group_name][attr_name][:], tensor)
    assert torch.equal(store[group_name].feat[:], tensor)
    assert torch.equal(store.view().A.feat[:], tensor)

    with pytest.raises(AttributeError) as exc_info:
        _ = store.view(group_name=group_name, index=None).feat.A
        print(exc_info)

    # Partially-specified forms, which produce an AttrView object
    assert store[group_name] == store.view(TensorAttr(group_name=group_name))
    assert store[group_name].feat == store.view(
        TensorAttr(group_name=group_name, attr_name=attr_name))

    # Partially-specified forms, when called, produce a Tensor output
    # from the `TensorAttr` that has been partially specified.
    store[group_name] = tensor
    assert isinstance(store[group_name], AttrView)
    assert torch.equal(store[group_name](), tensor)

    # Deletion:
    del store[group_name, attr_name, index]
    with pytest.raises(KeyError):
        _ = store[group_name, attr_name, index]
    del store[group_name]
    with pytest.raises(KeyError):
        _ = store[group_name]()


def test_feature_store_override():
    store = MyFeatureStoreNoGroupName()
    tensor = torch.Tensor([[0, 0, 0], [1, 1, 1], [2, 2, 2]])

    attr_name = 'feat'
    index = torch.tensor([0, 1, 2])

    # Only use attr_name and index, in that order:
    store[attr_name, index] = tensor

    # A few assertions to ensure group_name is not needed:
    assert isinstance(store[attr_name], AttrView)
    assert torch.equal(store[attr_name, index], tensor)
    assert torch.equal(store[attr_name][index], tensor)
    assert torch.equal(store[attr_name][:], tensor)
    assert torch.equal(store[attr_name, :], tensor)
