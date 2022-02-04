from dataclasses import dataclass

from torch_geometric.graphgym.config import from_config


@dataclass
class MyConfig:
    a: int
    b: int = 4


def my_func(a: int, b: int = 2) -> str:
    return f'a={a},b={b}'


def test_from_config():
    assert my_func(a=1) == 'a=1,b=2'

    assert my_func.__name__ == from_config(my_func).__name__
    assert from_config(my_func)(cfg=MyConfig(a=1)) == 'a=1,b=4'
    assert from_config(my_func)(cfg=MyConfig(a=1, b=1)) == 'a=1,b=1'
    assert from_config(my_func)(2, cfg=MyConfig(a=1, b=3)) == 'a=2,b=3'
    assert from_config(my_func)(cfg=MyConfig(a=1), b=3) == 'a=1,b=3'
