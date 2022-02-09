import inspect
from dataclasses import dataclass, field, make_dataclass
from typing import Any, Dict, List, Optional, Union

import torch
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

import torch_geometric.datasets as datasets
import torch_geometric.nn.models.basic_gnn as models
import torch_geometric.transforms as transforms
from torch_geometric.typing import map_annotation

EXCLUDE = {'self', 'args', 'kwargs'}

MAPPING = {
    torch.nn.Module: Any,
}


def to_dataclass(cls: Any, base: Optional[Any] = None,
                 exclude: Optional[List[str]] = None) -> Any:
    r"""Converts the input arguments of a given class :obj:`cls` to a
    :obj:`dataclass` schema, *e.g.*,

    .. code-block:: python

        from torch_geometric.transforms import NormalizeFeatures

        dataclass = to_dataclass(NormalizeFeatures)

    will generate

    .. code-block:: python

        @dataclass
        class NormalizeFeatures:
            _target_: str = "torch_geometric.transforms.NormalizeFeatures"
            attrs: List[str] = field(default_factory = lambda: ["x"])

    Args:
        cls (Any): The class to generate a schema for.
        base: (Any, optional): The base class of the schema.
            (default: :obj:`None`)
        exclude (List[str], optional): Arguments to exclude.
            (default: :obj:`None`)
    """
    fields = []

    for name, arg in inspect.signature(cls.__init__).parameters.items():
        if name in EXCLUDE:
            continue
        if exclude is not None and name in exclude:
            continue
        if base is not None and name in base.__dataclass_fields__.keys():
            continue

        item = (name, )

        annotation = arg.annotation
        default = arg.default

        annotation = map_annotation(annotation, mapping=MAPPING)

        if str(default) == "<required parameter>":
            # Fix `torch.optim.SGD.lr = _RequiredParameter()`
            default = inspect.Parameter.empty

        if annotation != inspect.Parameter.empty:
            # `Union` types are not supported (except for `Optional`).
            # As such, we replace them with either `Any` or `Optional[Any]`.
            origin = getattr(annotation, '__origin__', None)
            args = getattr(annotation, '__args__', [])
            if origin == Union and type(None) in args and len(args) > 2:
                annotation = Optional[Any]
            elif origin == Union and type(None) not in args:
                annotation = Any
        else:
            if default != inspect.Parameter.empty:
                annotation = Optional[Any]
            else:
                annotation = Any
        item = item + (annotation, )

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


config_store = ConfigStore.instance()


@dataclass
class BaseConfig:
    _target_: str = MISSING


@dataclass  # Register `torch_geometric.transforms` ###########################
class Transform(BaseConfig):
    pass


for cls_name in set(transforms.__all__) - set([
        'BaseTransform',
        'Compose',
        'LinearTransformation',
        'AddMetaPaths',  # TODO
]):
    cls = to_dataclass(getattr(transforms, cls_name), base=Transform)
    # We use an explicit additional nesting level inside each config to allow
    # for applying multiple transformations.
    # https://hydra.cc/docs/patterns/select_multiple_configs_from_config_group
    config_store.store(group='transform', name=cls_name, node={cls_name: cls})


@dataclass  # Register `torch_geometric.datasets` #############################
class Dataset(BaseConfig):
    transform: Dict[str, Transform] = field(default_factory=dict)
    pre_transform: Dict[str, Transform] = field(default_factory=dict)


for cls_name in set(datasets.__all__) - set([]):
    cls = to_dataclass(getattr(datasets, cls_name), base=Dataset,
                       exclude=['pre_filter'])
    config_store.store(group='dataset', name=cls_name, node=cls)


@dataclass  # Register `torch_geometric.models` ###############################
class Model(BaseConfig):
    pass


for cls_name in set(models.__all__) - set([]):
    cls = to_dataclass(getattr(models, cls_name), base=Model, exclude=[])
    config_store.store(group='model', name=cls_name, node=cls)


@dataclass  # Register `torch.optim.Optimizer` ################################
class Optimizer(BaseConfig):
    pass


for cls_name in set([
        key for key, cls in torch.optim.__dict__.items()
        if inspect.isclass(cls) and issubclass(cls, torch.optim.Optimizer)
]) - set(['Optimizer']):
    cls = to_dataclass(getattr(torch.optim, cls_name), base=Optimizer,
                       exclude=['params'])
    config_store.store(group='optimizer', name=cls_name, node=cls)


@dataclass  # Register `torch.optim.lr_scheduler` #############################
class LRScheduler(BaseConfig):
    pass


for cls_name in set([
        key for key, cls in torch.optim.lr_scheduler.__dict__.items()
        if inspect.isclass(cls)
]) - set([
        'Optimizer', '_LRScheduler', 'Counter', 'SequentialLR',
        'ChainedScheduler'
]):
    cls = to_dataclass(getattr(torch.optim.lr_scheduler, cls_name),
                       base=LRScheduler, exclude=['optimizer'])
    config_store.store(group='lr_scheduler', name=cls_name, node=cls)


@dataclass  # Register global schema ##########################################
class Config:
    dataset: Dataset = MISSING
    optim: Optimizer = MISSING
    lr_scheduler: Optional[LRScheduler] = None
