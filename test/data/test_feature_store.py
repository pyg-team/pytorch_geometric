from dataclasses import dataclass

import pytest
import torch

from torch_geometric.data import TensorAttr
from torch_geometric.data.feature_store import AttrView, _field_status
from torch_geometric.testing import MyFeatureStore


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
