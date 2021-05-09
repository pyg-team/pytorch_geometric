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

    assert len(list(storage.keys('x', 'y', 'z'))) == 2
    assert len(list(storage.keys('x', 'y', 'z'))) == 2
    assert len(list(storage.values('x', 'y', 'z'))) == 2
    assert len(list(storage.items('x', 'y', 'z'))) == 2

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

    storage = Storage(key='key', parent={}, x=torch.zeros(1))

    copied_storage = copy.copy(storage)
    assert storage == copied_storage
    assert id(storage) != id(copied_storage)
    assert id(storage._key) == id(copied_storage._key)
    assert id(storage._parent) == id(copied_storage._parent)
    assert storage.x.data_ptr() == copied_storage.x.data_ptr()
    assert int(storage.x) == 0
    assert int(copied_storage.x) == 0

    deepcopied_storage = copy.deepcopy(storage)
    assert storage == deepcopied_storage
    assert id(storage) != id(deepcopied_storage)
    assert id(storage._key) == id(deepcopied_storage._key)
    assert id(storage._parent) == id(deepcopied_storage._parent)
    assert storage.x.data_ptr() != deepcopied_storage.x.data_ptr()
    assert int(storage.x) == 0
    assert int(deepcopied_storage.x) == 0


def test_graph_storage():
    storage = Storage(x=1, y=2)
    # print(storage)
