import copy

import torch

from torch_geometric.data.storage import Storage


def test_storage():
    storage = Storage()
    storage.x = torch.zeros(1)
    storage.y = torch.ones(1)
    assert len(storage) == 2
    assert storage.x is not None
    assert storage.y is not None

    keys = storage.keys('x', 'y', 'z')
    assert len(keys) == 2

    values = storage.values('x', 'y', 'z')
    assert len(values) == 2

    items = storage.items('x', 'y', 'z')
    assert len(items) == 2

    del storage.y
    assert len(storage) == 1
    assert storage.x is not None
    assert storage.y is None

    storage = Storage({'x': torch.zeros(1)})
    assert len(storage) == 1
    assert storage.x is not None

    storage = Storage(x=torch.zeros(1))
    assert len(storage) == 1
    assert storage.x is not None

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


def test_storage_functionality():
    pass
