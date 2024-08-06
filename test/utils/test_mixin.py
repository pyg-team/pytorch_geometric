from torch_geometric.utils.mixin import CastMixin


class MyClass1(CastMixin):
    def __init__(self, x, y, z=42):
        self.x = x
        self.y = y
        self.z = z


class MyClass2(CastMixin):
    def __init__(self, a, b, c=42):
        self.a = a
        self.b = b
        self.c = c


def test_cast_mixin():
    obj1 = MyClass1(1, 2, z=3)
    obj2 = MyClass2.cast(obj1)
    assert isinstance(obj2, MyClass2)
    assert obj2.a == 1
    assert obj2.b == 2
    assert obj2.c == 3
