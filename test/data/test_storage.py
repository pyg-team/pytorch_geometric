import copy

import torch

from torch_geometric.data.storage import Storage, Storage2


def test_storage():
    storage = Storage2()
    storage.x = torch.zeros(1)
    storage.y = torch.ones(1)
    assert len(storage) == 2
    assert storage.x is not None
    assert storage.y is not None

    # keys = storage.keys('x', 'y', 'z')
    # assert len(keys) == 2

    # values = storage.values('x', 'y', 'z')
    # assert len(values) == 2

    # items = storage.items('x', 'y', 'z')
    # assert len(items) == 2

    del storage.y
    assert len(storage) == 1
    assert storage.x is not None
    assert storage.y is None

    storage = Storage2({'x': torch.zeros(1)})
    assert len(storage) == 1
    assert storage.x is not None

    storage = Storage2(x=torch.zeros(1))
    assert len(storage) == 1
    assert storage.x is not None

    storage = Storage2(key='key', parent={}, x=torch.zeros(1))

    copied_storage = copy.copy(storage)
    assert storage == copied_storage
    assert id(storage) != id(copied_storage)
    assert id(storage._key) == id(storage._key)
    assert id(storage._parent) == id(storage._parent)
    assert storage.x.data_ptr() == copied_storage.x.data_ptr()
    assert int(storage.x) == 0
    assert int(copied_storage.x) == 0

    deepcopied_storage = copy.deepcopy(storage)
    assert storage == deepcopied_storage
    assert id(storage) != id(deepcopied_storage)
    assert id(storage._key) == id(storage._key)
    assert id(storage._parent) == id(storage._parent)
    assert storage.x.data_ptr() != deepcopied_storage.x.data_ptr()
    assert int(storage.x) == 0
    assert int(deepcopied_storage.x) == 0


def test_storage_functionality():
    storage = Storage2(x=1, y=2)
    bla = storage.items()
    # print(storage)
    # print(bla)
