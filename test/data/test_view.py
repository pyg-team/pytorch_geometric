from torch_geometric.data.view import (
    ItemsView,
    KeysView,
    MappingView,
    ValuesView,
)

class M:
    def keys(self):
        return [1, 2, 3]

    def __getitem__(self, x):
        return x * 2

def test_mapping_view():
    assert M().keys() == [1, 2, 3]
    d = dict(M())
    assert d == {1: 2, 2: 4, 3: 6}
    mview = MappingView(mapping=M())
    assert mview._keys() == [1, 2, 3]
    assert mview.__len__() == 3

def test_keys_view():
    kview = KeysView(mapping=M())
    for i, val in enumerate(kview.__iter__()):
        assert i + 1 == val

def test_values_view():
    vview = ValuesView(mapping=M())
    for i, val in enumerate(vview.__iter__()):
        assert 2 * (i + 1) == val


def test_items_view():
    iview = ItemsView(mapping=M())
    for i, val in enumerate(iview.__iter__()):
        assert val == (i + 1, 2 * i + 2)
