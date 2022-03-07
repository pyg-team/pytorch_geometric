import inspect
import re
from dataclasses import dataclass, field, make_dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from omegaconf import MISSING

from torch_geometric.typing import map_annotation

EXCLUDE = {'self', 'args', 'kwargs'}

MAPPING = {
    torch.nn.Module: Any,
}


def get_config_store() -> Any:
    r"""Return the global configuration store."""
    from hydra.core.config_store import ConfigStore
    return ConfigStore.instance()


def to_dataclass(
    cls: Any,
    base: Optional[Any] = None,
    map_args: Optional[Dict[str, Tuple]] = None,
    exclude_args: Optional[List[str]] = None,
    strict: bool = False,
) -> Any:
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
        map_args (Dict[str, Tuple], optional): Arguments for which annotation
            and default values should be overriden. (default: :obj:`None`)
        exclude_args (List[str or int], optional): Arguments to exclude.
            (default: :obj:`None`)
        strict (bool, optional): If set to :obj:`True`, ensures that all
            arguments in either :obj:`map_args` and :obj:`exclude_args` are
            present in the input parameters. (default: :obj:`False`)
    """
    fields = []

    params = inspect.signature(cls.__init__).parameters

    if strict:  # Check that keys in map_args or exclude_args are present.
        args = set() if map_args is None else set(map_args.keys())
        if exclude_args is not None:
            args |= set([arg for arg in exclude_args if isinstance(arg, str)])
        diff = args - set(params.keys())
        if len(diff) > 0:
            raise ValueError(f"Expected input argument(s) {diff} in "
                             f"'{cls.__name__}'")

    for i, (name, arg) in enumerate(params.items()):
        if name in EXCLUDE:
            continue
        if exclude_args is not None:
            if name in exclude_args or i in exclude_args:
                continue
        if base is not None:
            if name in base.__dataclass_fields__:
                continue

        if map_args is not None and name in map_args:
            fields.append((name, ) + map_args[name])
            continue

        annotation, default = arg.annotation, arg.default
        annotation = map_annotation(annotation, mapping=MAPPING)

        if annotation != inspect.Parameter.empty:
            # `Union` types are not supported (except for `Optional`).
            # As such, we replace them with either `Any` or `Optional[Any]`.
            origin = getattr(annotation, '__origin__', None)
            args = getattr(annotation, '__args__', [])
            if origin == Union and type(None) in args and len(args) > 2:
                annotation = Optional[Any]
            elif origin == Union and type(None) not in args:
                annotation = Any
            elif origin == list:
                if getattr(args[0], '__origin__', None) == Union:
                    annotation = List[Any]
            elif origin == dict:
                if getattr(args[1], '__origin__', None) == Union:
                    annotation = Dict[args[0], Any]
        else:
            if default != inspect.Parameter.empty:
                annotation = Optional[Any]
            else:
                annotation = Any

        if str(default) == "<required parameter>":
            # Fix `torch.optim.SGD.lr = _RequiredParameter()`:
            # https://github.com/pytorch/hydra-torch/blob/main/
            # hydra-configs-torch/hydra_configs/torch/optim/sgd.py
            default = field(default=MISSING)
        elif default != inspect.Parameter.empty:
            if isinstance(default, (list, dict)):
                # Avoid late binding of default values inside a loop:
                # https://stackoverflow.com/questions/3431676/
                # creating-functions-in-a-loop
                def wrapper(default):
                    return lambda: default

                default = field(default_factory=wrapper(default))
        else:
            default = field(default=MISSING)

        fields.append((name, annotation, default))

    full_cls_name = f'{cls.__module__}.{cls.__qualname__}'
    fields.append(('_target_', str, field(default=full_cls_name)))

    return make_dataclass(cls.__qualname__, fields=fields,
                          bases=() if base is None else (base, ))


def register(
    group: str,
    cls: Optional[Any] = None,
    **kwargs,
) -> Optional[Callable]:
    r"""Registers a class in the global configuration store.

    Args:
        group (str): The group of the config
        cls (cls, optional): The class to register. If set to :obj:`None`, will
            return a decorator. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`~torch_geometric.graphgym.config_store.to_dataclass`.
    """

    if cls is not None:
        name = cls.__name__
        cls = to_dataclass(cls, **kwargs)

        get_config_store().store(name, cls, group)

        pattern = re.compile(group, re.IGNORECASE)
        if pattern.search(name):
            get_config_store().store(pattern.sub('', name), cls, group)
        return None

    def bounded_register(cls: Any) -> Any:  # Other-wise, use it as a decorator
        register(group, cls, **kwargs)
        return cls

    return bounded_register


@dataclass
class Transform:
    pass


@dataclass
class Dataset:
    pass


@dataclass
class Model:
    pass


@dataclass
class Optimizer:
    pass


@dataclass
class LRScheduler:
    pass


@dataclass
class Config:
    dataset: Dataset = MISSING
    model: Model = MISSING
    optim: Optimizer = MISSING
    lr_scheduler: Optional[LRScheduler] = None


def fill_config_store():
    import torch_geometric

    config_store = get_config_store()

    # Register `torch_geometric.transforms` ###################################
    transforms = torch_geometric.transforms
    for cls_name in set(transforms.__all__) - set([
            'BaseTransform',
            'Compose',
            'LinearTransformation',
            'AddMetaPaths',  # TODO
    ]):
        cls = to_dataclass(getattr(transforms, cls_name), base=Transform)
        # We use an explicit additional nesting level inside each config to
        # allow for applying multiple transformations.
        # See: hydra.cc/docs/patterns/select_multiple_configs_from_config_group
        config_store.store(cls_name, group='transform', node={cls_name: cls})

    # Register `torch_geometric.datasets` #####################################
    datasets = torch_geometric.datasets
    map_dataset_args = {
        'transform': (Dict[str, Transform], field(default_factory=dict)),
        'pre_transform': (Dict[str, Transform], field(default_factory=dict)),
    }

    for cls_name in set(datasets.__all__) - set([]):
        cls = to_dataclass(getattr(datasets, cls_name), base=Dataset,
                           map_args=map_dataset_args,
                           exclude_args=['pre_filter'])
        config_store.store(cls_name, group='dataset', node=cls)

    # Register `torch_geometric.models` #######################################
    models = torch_geometric.nn.models.basic_gnn
    for cls_name in set(models.__all__) - set([]):
        cls = to_dataclass(getattr(models, cls_name), base=Model)
        config_store.store(cls_name, group='model', node=cls)

    # Register `torch.optim.Optimizer` ########################################
    for cls_name in set([
            key for key, cls in torch.optim.__dict__.items()
            if inspect.isclass(cls) and issubclass(cls, torch.optim.Optimizer)
    ]) - set([
            'Optimizer',
    ]):
        cls = to_dataclass(getattr(torch.optim, cls_name), base=Optimizer,
                           exclude_args=['params'])
        config_store.store(cls_name, group='optimizer', node=cls)

    # Register `torch.optim.lr_scheduler` #####################################
    for cls_name in set([
            key for key, cls in torch.optim.lr_scheduler.__dict__.items()
            if inspect.isclass(cls)
    ]) - set([
            'Optimizer',
            '_LRScheduler',
            'Counter',
            'SequentialLR',
            'ChainedScheduler',
    ]):
        cls = to_dataclass(getattr(torch.optim.lr_scheduler, cls_name),
                           base=LRScheduler, exclude_args=['optimizer'])
        config_store.store(cls_name, group='lr_scheduler', node=cls)

    # Register global schema ##################################################
    config_store.store('config', node=Config)
