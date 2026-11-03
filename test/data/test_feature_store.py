from dataclasses import dataclass

import pytest
import torch

from torch_geometric.data import TensorAttr
from torch_geometric.data.feature_store import AttrView, _FieldStatus
from torch_geometric.testing import MyFeatureStore


@dataclass
class MyTensorAttrNoGroupName(TensorAttr):
    def __init__(self, attr_name=_FieldStatus.UNSET, index=_FieldStatus.UNSET):
        # Treat group_name as optional, and move it to the end
        super().__init__(None, attr_name, index)


class MyFeatureStoreNoGroupName(MyFeatureStore):
    def __init__(self):
        super().__init__()
        self._tensor_attr_cls = MyTensorAttrNoGroupName


def test_feature_store():
    store = MyFeatureStore()
    tensor = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])

    group_name = 'A'
    attr_name = 'feat'
    index = torch.tensor([0, 1, 2])
    attr = TensorAttr(group_name, attr_name, index)
    assert TensorAttr(group_name).update(attr) == attr

    # Normal API:
    store.put_tensor(tensor, attr)
    assert torch.equal(store.get_tensor(attr), tensor)
    assert torch.equal(
        store.get_tensor(group_name, attr_name, index=torch.tensor([0, 2])),
        tensor[torch.tensor([0, 2])],
    )

    assert store.update_tensor(tensor + 1, attr)
    assert torch.equal(store.get_tensor(attr), tensor + 1)

    store.remove_tensor(attr)
    with pytest.raises(KeyError):
        _ = store.get_tensor(attr)

    # Views:
    view = store.view(group_name=group_name)
    view.attr_name = attr_name
    view['index'] = index
    assert view != "not a 'AttrView' object"
    assert view == AttrView(store, TensorAttr(group_name, attr_name, index))
    assert str(view) == ("AttrView(store=MyFeatureStore(), "
                         "attr=TensorAttr(group_name='A', attr_name='feat', "
                         "index=tensor([0, 1, 2])))")

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
    store[group_name, None, None] = tensor
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
    tensor = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])

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


@pytest.mark.parametrize('attr_cls', [TensorAttr, MyTensorAttrNoGroupName])
def test_tensor_attr_fully_specify(attr_cls):
    attr = attr_cls(attr_name='x')
    assert attr.fully_specify() is attr
    assert attr.is_fully_specified()
    assert attr.group_name is None
    assert attr.attr_name == 'x'
    assert attr.index is None
    assert attr.fully_specify() is attr

    index = torch.tensor([1, 2])
    attr = attr_cls(attr_name='x', index=index)
    attr.fully_specify()
    assert attr.index is index


@pytest.mark.parametrize('store_cls', [MyFeatureStore, MyFeatureStoreNoGroupName])
def test_feature_store_partial_assignment(store_cls):
    store = store_cls()
    tensor = torch.arange(6).view(3, 2)
    key = ('paper', 'x') if store_cls is MyFeatureStore else ('x',)
    store[key] = tensor
    assert torch.equal(store[key + (None,)], tensor)
    assert torch.equal(store[key](), tensor)
    assert not store[key]._attr.is_set('index')
    assert torch.equal(store[key + (torch.tensor([2, 0]),)], tensor[[2, 0]])
