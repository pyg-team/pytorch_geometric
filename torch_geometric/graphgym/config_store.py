import copy
import inspect
from collections import defaultdict
from dataclasses import dataclass, field, make_dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch

EXCLUDE = {'self', 'args', 'kwargs'}

MAPPING = {
    torch.nn.Module: Any,
    torch.Tensor: Any,
}

try:
    from omegaconf import MISSING
except ImportError:
    MISSING: Any = '???'

try:
    from hydra.core.config_store import ConfigStore

    def get_node(cls: Union[str, Any]) -> Optional[Any]:
        if (not isinstance(cls, str)
                and cls.__module__ in {'builtins', 'typing'}):
            return None

        def _recursive_get_node(repo: Dict[str, Any]) -> Optional[Any]:
            for key, value in repo.items():
                if isinstance(value, dict):
                    out = _recursive_get_node(value)
                    if out is not None:
                        return out
                elif isinstance(cls, str) and key == f'{cls}.yaml':
                    return value.node
                elif getattr(value.node._metadata, 'object_type', None) == cls:
                    return value.node
                elif getattr(value.node._metadata, 'orig_type', None) == cls:
                    return value.node

        return _recursive_get_node(get_config_store().repo)

    def dataclass_from_class(cls: Union[str, Any]) -> Optional[Any]:
        r"""Returns the :obj:`dataclass` of a class registered in the global
        configuration store."""
        node = get_node(cls)
        return node._metadata.object_type if node is not None else None

    def class_from_dataclass(cls: Union[str, Any]) -> Optional[Any]:
        r"""Returns the original class of a :obj:`dataclass` registered in the
        global configuration store."""
        node = get_node(cls)
        return node._metadata.orig_type if node is not None else None

except ImportError:

    class Singleton(type):
        _instances: Dict[type, Any] = {}

        def __call__(cls, *args, **kwargs) -> Any:
            if cls not in cls._instances:
                instance = super(Singleton, cls).__call__(*args, **kwargs)
                cls._instances[cls] = instance
                return instance
            return cls._instances[cls]

    @dataclass
    class Metadata:
        orig_type: Optional[Any] = None

    @dataclass
    class ConfigNode:
        name: str
        node: Any
        group: Optional[str] = None
        _metadata: Metadata = field(default_factory=Metadata)

    class ConfigStore(metaclass=Singleton):
        def __init__(self):
            self.repo: Dict[str, Any] = defaultdict(dict)

        @classmethod
        def instance(cls, *args, **kwargs) -> 'ConfigStore':
            return cls(*args, **kwargs)

        def store(self, name: str, node: Any, group: Optional[str] = None):
            cur = self.repo
            if group is not None:
                cur = cur[group]
            cur[name] = ConfigNode(name, node, group)

    def get_node(cls: Union[str, Any]) -> Optional[ConfigNode]:
        if (not isinstance(cls, str)
                and cls.__module__ in {'builtins', 'typing'}):
            return None

        def _recursive_get_node(repo: Dict[str, Any]) -> Optional[ConfigNode]:
            for key, value in repo.items():
                if isinstance(value, dict):
                    out = _recursive_get_node(value)
                    if out is not None:
                        return out
                elif isinstance(cls, str) and key == cls:
                    return value
                elif value.node == cls:
                    return value
                elif value._metadata.orig_type == cls:
                    return value

        return _recursive_get_node(get_config_store().repo)

    def dataclass_from_class(cls: Union[str, Any]) -> Optional[Any]:
        r"""Returns the :obj:`dataclass` of a class registered in the global
        configuration store."""
        node = get_node(cls)
        return node.node if node is not None else None

    def class_from_dataclass(cls: Union[str, Any]) -> Optional[Any]:
        r"""Returns the original class of a :obj:`dataclass` registered in the
        global configuration store."""
        node = get_node(cls)
        return node._metadata.orig_type if node is not None else None


def map_annotation(
    annotation: Any,
    mapping: Optional[Dict[Any, Any]] = None,
) -> Any:

    origin = getattr(annotation, '__origin__', None)
    args = getattr(annotation, '__args__', [])
    if origin == Union or origin == list or origin == dict:
        annotation = copy.copy(annotation)
        annotation.__args__ = tuple(map_annotation(a, mapping) for a in args)
        return annotation

    if annotation in mapping or {}:
        return mapping[annotation]

    out = dataclass_from_class(annotation)
    if out is not None:
        return out

    return annotation


def to_dataclass(
    cls: Any,
    base_cls: Optional[Any] = None,
    with_target: Optional[bool] = None,
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
        base_cls (Any, optional): The base class of the schema.
            (default: :obj:`None`)
        with_target (bool, optional): If set to :obj:`False`, will not add the
            :obj:`_target_` attribute to the schema. If set to :obj:`None`,
            will only add the :obj:`_target_` in case :obj:`base_cls` is given.
            (default: :obj:`None`)
        map_args (Dict[str, Tuple], optional): Arguments for which annotation
            and default values should be overridden. (default: :obj:`None`)
        exclude_args (List[str or int], optional): Arguments to exclude.
            (default: :obj:`None`)
        strict (bool, optional): If set to :obj:`True`, ensures that all
            arguments in both :obj:`map_args` and :obj:`exclude_args` are
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
        if base_cls is not None:
            if name in base_cls.__dataclass_fields__:
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

    with_target = base_cls is not None if with_target is None else with_target
    if with_target:
        full_cls_name = f'{cls.__module__}.{cls.__qualname__}'
        fields.append(('_target_', str, field(default=full_cls_name)))

    return make_dataclass(cls.__qualname__, fields=fields,
                          bases=() if base_cls is None else (base_cls, ))


def get_config_store() -> ConfigStore:
    r"""Returns the global configuration store."""
    return ConfigStore.instance()


def register(
    cls: Optional[Any] = None,
    data_cls: Optional[Any] = None,
    group: Optional[str] = None,
    **kwargs,
) -> Union[Any, Callable]:
    r"""Registers a class in the global configuration store.

    Args:
        cls (Any, optional): The class to register. If set to :obj:`None`, will
            return a decorator. (default: :obj:`None`)
        data_cls (Any, optional): The data class to register. If set to
            :obj:`None`, will dynamically create the data class according to
            :class:`~torch_geometric.graphgym.config_store.to_dataclass`.
            (default: :obj:`None`)
        group (str, optional): The group in the global configuration store.
            (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`~torch_geometric.graphgym.config_store.to_dataclass`.
    """
    if cls is not None:
        name = cls.__name__

        if data_cls is None:
            data_cls = to_dataclass(cls, **kwargs)

        get_config_store().store(name, data_cls, group)
        get_node(name)._metadata.orig_type = cls

        return data_cls

    def bounded_register(cls: Any) -> Any:  # Other-wise, return a decorator:
        register(cls=cls, data_cls=data_cls, group=group, **kwargs)
        return cls

    return bounded_register


###############################################################################


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
            'ComposeFilters',
            'LinearTransformation',
            'AddMetaPaths',  # TODO
    ]):
        cls = to_dataclass(getattr(transforms, cls_name), base_cls=Transform)
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
        cls = to_dataclass(getattr(datasets, cls_name), base_cls=Dataset,
                           map_args=map_dataset_args,
                           exclude_args=['pre_filter'])
        config_store.store(cls_name, group='dataset', node=cls)

    # Register `torch_geometric.models` #######################################
    models = torch_geometric.nn.models.basic_gnn
    for cls_name in set(models.__all__) - set([]):
        cls = to_dataclass(getattr(models, cls_name), base_cls=Model)
        config_store.store(cls_name, group='model', node=cls)

    # Register `torch.optim.Optimizer` ########################################
    for cls_name in set([
            key for key, cls in torch.optim.__dict__.items()
            if inspect.isclass(cls) and issubclass(cls, torch.optim.Optimizer)
    ]) - set([
            'Optimizer',
    ]):
        cls = to_dataclass(getattr(torch.optim, cls_name), base_cls=Optimizer,
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
                           base_cls=LRScheduler, exclude_args=['optimizer'])
        config_store.store(cls_name, group='lr_scheduler', node=cls)

    # Register global schema ##################################################
    config_store.store('config', node=Config)
