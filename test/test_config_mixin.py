from dataclasses import asdict, dataclass, is_dataclass
from typing import Sequence

import pytest
import torch

from torch_geometric.config_mixin import ConfigMixin
from torch_geometric.config_store import clear_config_store, register


@pytest.fixture(scope="session", autouse=True)
def teardown_once():
    yield  # This allows tests to run before teardown is executed
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


@register(with_target=True)
class SubModule(Base):
    def __init__(self, p: float):
        super().__init__()
        self.p = p


@register(with_target=True)
class CompoundModule(torch.nn.Module, ConfigMixin):
    def __init__(self, z: int, module: Module, submodules: list[SubModule]):
        super().__init__()
        self.z = z
        self.module = module
        self.submodules = torch.nn.ModuleList(submodules)


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
    assert cfg._target_ == 'test.test_config_mixin.Module'

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


def test_config_mixin_compound() -> None:
    submodules = [SubModule(1.41), SubModule(3.14)]
    module = Module(x=0, data=Dataclass(x=1, y=2))
    model = CompoundModule(z=3, module=module, submodules=submodules)
    cfg = model.config()
    assert is_dataclass(cfg)
    assert cfg._target_ == 'test_config_mixin.CompoundModule'
    assert cfg.z == 3
    assert cfg.module._target_ == 'test_config_mixin.Module'
    assert cfg.module.x == 0
    assert isinstance(cfg.module.data, Dataclass)
    assert cfg.module.data.x == 1
    assert cfg.module.data.y == 2
    assert len(cfg.submodules) == 2
    assert isinstance(cfg.submodules, Sequence)
    assert cfg.submodules[0]._target_ == 'test_config_mixin.SubModule'
    assert cfg.submodules[0].p == 1.41
    assert cfg.submodules[1]._target_ == 'test_config_mixin.SubModule'
    assert cfg.submodules[1].p == 3.14

    model = CompoundModule.from_config(cfg)
    assert isinstance(model, CompoundModule)
    assert model.z == 3
    assert isinstance(model.module, Module)
    assert model.module.x == 0
    assert isinstance(model.module.data, Dataclass)
    assert model.module.data.x == 1
    assert model.module.data.y == 2
    assert isinstance(model.submodules, torch.nn.ModuleList)
    assert len(model.submodules) == 2
    assert isinstance(model.submodules[0], SubModule)
    assert model.submodules[0].p == 1.41
    assert isinstance(model.submodules[1], SubModule)
    assert model.submodules[1].p == 3.14
