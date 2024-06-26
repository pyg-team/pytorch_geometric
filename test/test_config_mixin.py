from dataclasses import asdict, dataclass, is_dataclass

import torch

from torch_geometric.config_mixin import ConfigMixin
from torch_geometric.config_store import clear_config_store, register


def teardown_function() -> None:
    clear_config_store()


@dataclass
class Dataclass:
    x: int
    y: int


class Base(torch.nn.Module, ConfigMixin):
    pass


@register(with_target=True)
class Module(Base):
    def __init__(self, x: int, data: Dataclass):
        super().__init__()
        self.x = x
        self.data = data


def test_config_mixin() -> None:
    x = 0
    data = Dataclass(x=1, y=2)

    model = Module(x, data)
    cfg = model.config()
    assert is_dataclass(cfg)
    assert cfg.x == 0
    assert isinstance(cfg.data, Dataclass)
    assert cfg.data.x == 1
    assert cfg.data.y == 2
    assert cfg._target_ == 'test_config_mixin.Module'

    model = Module.from_config(cfg)
    assert isinstance(model, Module)
    assert model.x == 0
    assert isinstance(model.data, Dataclass)
    assert model.data.x == 1
    assert model.data.y == 2

    model = Base.from_config(cfg)
    assert isinstance(model, Module)
    assert model.x == 0
    assert isinstance(model.data, Dataclass)
    assert model.data.x == 1
    assert model.data.y == 2

    model = Base.from_config(cfg, 3)
    assert isinstance(model, Module)
    assert model.x == 3
    assert isinstance(model.data, Dataclass)
    assert model.data.x == 1
    assert model.data.y == 2

    model = Base.from_config(cfg, data=Dataclass(x=2, y=3))
    assert isinstance(model, Module)
    assert model.x == 0
    assert isinstance(model.data, Dataclass)
    assert model.data.x == 2
    assert model.data.y == 3

    cfg = asdict(cfg)

    model = Module.from_config(cfg)
    assert isinstance(model, Module)
    assert model.x == 0
    assert isinstance(model.data, dict)
    assert model.data['x'] == 1
    assert model.data['y'] == 2

    model = Base.from_config(cfg)
    assert isinstance(model, Module)
    assert model.x == 0
    assert isinstance(model.data, dict)
    assert model.data['x'] == 1
    assert model.data['y'] == 2
