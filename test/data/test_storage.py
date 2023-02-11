import copy
from typing import Any

import torch

from torch_geometric.data.storage import BaseStorage, NodeStorage
from torch_geometric.testing import onlyCUDA


def test_base_storage():
    storage = BaseStorage()
    storage.x = torch.zeros(1)
    storage.y = torch.ones(1)
    assert len(storage) == 2
    assert storage.x is not None
    assert storage.y is not None

    assert torch.allclose(storage.get('x', None), storage.x)
    assert torch.allclose(storage.get('y', None), storage.y)
    assert storage.get('z', 2) == 2
    assert storage.get('z', None) is None
    assert getattr(storage, '_mapping') == {
        'x': torch.zeros(1),
        'y': torch.ones(1)
    }
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

    deepcopied_storage = storage.clone()
    assert storage == deepcopied_storage
    assert id(storage) != id(deepcopied_storage)
    assert storage.x.data_ptr() != deepcopied_storage.x.data_ptr()
    assert int(storage.x) == 0
    assert int(deepcopied_storage.x) == 0

    storage.share_memory_()
    storage.detach()
    storage.detach_()


@onlyCUDA
def test_cuda_storage():
    storage = BaseStorage()
    storage.x = torch.zeros(1)
    storage.y = torch.ones(1)
    storage.to("cuda:0")
    storage.cpu()
    storage.pin_memory()  # only dense CPU tensors can be pinned
    storage.cuda(0)


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
