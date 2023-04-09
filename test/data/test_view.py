from torch_geometric.data.storage import BaseStorage


def test_views():
    storage = BaseStorage(x=1, y=2, z=3)

    assert str(storage.keys()) == "KeysView({'x': 1, 'y': 2, 'z': 3})"
    assert len(storage.keys()) == 3
    assert list(storage.keys()) == ['x', 'y', 'z']

    assert str(storage.values()) == "ValuesView({'x': 1, 'y': 2, 'z': 3})"
    assert len(storage.values()) == 3
    assert list(storage.values()) == [1, 2, 3]

    assert str(storage.items()) == "ItemsView({'x': 1, 'y': 2, 'z': 3})"
    assert len(storage.items()) == 3
    assert list(storage.items()) == [('x', 1), ('y', 2), ('z', 3)]

    args = ['x', 'z', 'foo']

    assert str(storage.keys(*args)) == "KeysView({'x': 1, 'z': 3})"
    assert len(storage.keys(*args)) == 2
    assert list(storage.keys(*args)) == ['x', 'z']

    assert str(storage.values(*args)) == "ValuesView({'x': 1, 'z': 3})"
    assert len(storage.values(*args)) == 2
    assert list(storage.values(*args)) == [1, 3]

    assert str(storage.items(*args)) == "ItemsView({'x': 1, 'z': 3})"
    assert len(storage.items(*args)) == 2
    assert list(storage.items(*args)) == [('x', 1), ('z', 3)]
