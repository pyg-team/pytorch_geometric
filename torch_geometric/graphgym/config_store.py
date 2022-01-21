import inspect
from dataclasses import dataclass, field, make_dataclass
from typing import Any, List, Optional, Union, Dict, Tuple, Callable

from hydra.core.config_store import ConfigStore
import copy
from omegaconf import MISSING

import torch_geometric.transforms as transforms
import torch_geometric.datasets as datasets


def map_annotation(annotation: Any, mapping: Dict[Any, Any]) -> Any:
    if annotation.__dict__.get('__origin__', None) == Union:
        args = annotation.__dict__.get('__args__', [])
        out = copy.copy(annotation)
        out.__dict__['__args__'] = [map_annotation(a, mapping) for a in args]
        return out
    elif annotation in mapping:
        return mapping[annotation]
    return annotation


def to_dataclass(cls: Any, base: Optional[Any] = None, mapping=None) -> Any:
    r"""Converts a given class :obj:`cls` to a `dataclass` schema."""
    fields = []
    for name, parameter in inspect.signature(cls.__init__).parameters.items():
        if name in {'self', 'args', 'kwargs'}:
            continue

        item = (name, )

        annotation = parameter.annotation
        if annotation != inspect.Parameter.empty:
            print('1', annotation)
            if mapping is not None:
                annotation = map_annotation(annotation, mapping)
            # `Union` types are not supported (except `Optional`).
            # As such, we replace them with either `Any` or `Optional[Any]`.
            origin = annotation.__dict__.get('__origin__', None)
            args = annotation.__dict__.get('__args__', [])
            if origin == Union and type(None) in args and len(args) > 2:
                annotation = Optional[Any]
            elif origin == Union and type(None) not in args:
                annotation = Any
            print('2', annotation)

            item = item + (annotation, )
        else:
            item = item + (Any, )

        default = parameter.default
        if default != inspect.Parameter.empty:
            if isinstance(default, (list, dict)):
                item = item + (field(default_factory=lambda: default), )
            else:
                item = item + (default, )
        else:
            item = item + (field(default=MISSING), )

        fields.append(item)

    full_cls_name = f'{cls.__module__}.{cls.__qualname__}'
    fields.append(('_target_', str, field(default=full_cls_name)))

    return make_dataclass(cls.__qualname__, fields=fields,
                          bases=() if base is None else (base, ))


def register(group: str, package: Any, exclude: List[str] = []):
    for name in set(package.__all__) - set(exclude):
        Config = to_dataclass(getattr(package, name))
        config_store.store(group=group, name=name, node=Config)


config_store = ConfigStore.instance()


@dataclass
class Config:
    dataset: Any = MISSING


# Register base dataclasses:
bases: Dict[str, Any] = {}
Config = to_dataclass(getattr(transforms, 'BaseTransform'))
print(Config)
print(Config())
bases['BaseTransform'] = Config

mapping = {
    Callable: Any,
    transforms.BaseTransform: bases['BaseTransform'],
}

print(map_annotation(Optional[Callable], mapping))
print(map_annotation(Optional[transforms.BaseTransform], mapping))
print('-------------')

config_store.store(name='config', node=Config)

base = bases['BaseTransform']
Config = to_dataclass(getattr(transforms, 'AddSelfLoops'), base=base,
                      mapping=mapping)
config_store.store(group='transform', name='AddSelfLoops', node=Config)
print(Config)
print(Config())
print(Config.__bases__)

# print(config_store.load('transform'))
# print(config_store.repo['transform'])

# print(config_store.repo)
# print(config_store.repo.keys())
# print(config_store.repo['transform'].keys())

# config_node = config_store.load('transform/BaseTransform.yaml')
# print(config_node)
# print(config_node)

# Register `torch_geometric.transforms.*`:
# register(group='transform', package=torch_geometric.transforms,
#          exclude=['BaseTransform', 'AddMetaPaths'])

Config = to_dataclass(getattr(datasets, 'KarateClub'), mapping=mapping)
config_store.store(group='dataset', name='KarateClub', node=Config)

print(inspect.signature(Config))

# Config = to_dataclass(getattr(torch_geometric.datasets, 'Planetoid'))
# config_store.store(group='dataset', name='Planetoid', node=Config)

# # Register `torch_geometric.transforms.*`:
# # register(group='dataset', package=torch_geometric.datasets)
