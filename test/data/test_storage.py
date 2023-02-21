import copy
from typing import Any

import torch

from torch_geometric.data.storage import BaseStorage


def test_base_storage():
    storage = BaseStorage()
    assert storage._mapping == {}
    storage.x = torch.zeros(1)
    storage.y = torch.ones(1)
    assert len(storage) == 2
    assert storage._mapping == {'x': torch.zeros(1), 'y': torch.ones(1)}
    assert storage.x is not None
    assert storage.y is not None

    assert torch.allclose(storage.get('x', None), storage.x)
    assert torch.allclose(storage.get('y', None), storage.y)
    assert storage.get('z', 2) == 2
    assert storage.get('z', None) is None
    assert len(list(storage.keys('x', 'y', 'z'))) == 2
    assert len(list(storage.keys('x', 'y', 'z'))) == 2
    assert len(list(storage.values('x', 'y', 'z'))) == 2
    assert len(list(storage.items('x', 'y', 'z'))) == 2

    del storage.y
    assert len(storage) == 1
    assert storage.x is not None

    storage = BaseStorage({'x': torch.zeros(1)})
    assert len(storage) == 1
    assert storage.x is not None

    storage = BaseStorage(x=torch.zeros(1))
    assert len(storage) == 1
    assert storage.x is not None

    storage = BaseStorage(x=torch.zeros(1))
    copied_storage = copy.copy(storage)
    assert storage == copied_storage
    assert id(storage) != id(copied_storage)
    assert storage.x.data_ptr() == copied_storage.x.data_ptr()
    assert int(storage.x) == 0
    assert int(copied_storage.x) == 0

    deepcopied_storage = copy.deepcopy(storage)
    assert storage == deepcopied_storage
    assert id(storage) != id(deepcopied_storage)
    assert storage.x.data_ptr() != deepcopied_storage.x.data_ptr()
    assert int(storage.x) == 0
    assert int(deepcopied_storage.x) == 0


def test_storage_tensor_methods():
    x = torch.randn(5)
    storage = BaseStorage({'x': x})

    storage = storage.clone()
    assert storage.x.data_ptr() != x.data_ptr()

    storage = storage.contiguous()
    assert storage.x.is_contiguous()

    storage = storage.to('cpu')
    assert storage.x.device == torch.device('cpu')

    storage = storage.cpu()
    assert storage.x.device == torch.device('cpu')

    if torch.cuda.is_available():
        storage = storage.pin_memory()
        assert storage.x.is_pinned()

    storage = storage.share_memory_()
    assert storage.x.is_shared

    storage = storage.detach_()
    assert not storage.x.requires_grad

    storage = storage.detach()
    assert not storage.x.requires_grad

    storage = storage.requires_grad_()
    assert storage.x.requires_grad


def test_setter_and_getter():
    class MyStorage(BaseStorage):
        @property
        def my_property(self) -> Any:
            return self._my_property

        @my_property.setter
        def my_property(self, value: Any):
            self._my_property = value

    storage = MyStorage()
    storage.my_property = 'hello'
    assert storage.my_property == 'hello'
    assert storage._my_property == storage._my_property
