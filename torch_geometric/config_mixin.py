import inspect
from dataclasses import fields, is_dataclass
from importlib import import_module
from typing import Any, Dict

from torch.nn import ModuleDict, ModuleList

from torch_geometric.config_store import (
    class_from_dataclass,
    dataclass_from_class,
)
from torch_geometric.isinstance import is_torch_instance


class ConfigMixin:
    r"""Enables a class to serialize/deserialize itself to a dataclass."""
    def config(self) -> Any:
        r"""Creates a serializable configuration of the class."""
        data_cls = dataclass_from_class(self.__class__)
        if data_cls is None:
            raise ValueError(f"Could not find the configuration class that "
                             f"belongs to '{self.__class__.__name__}'. Please "
                             f"register it in the configuration store.")

        kwargs: Dict[str, Any] = {}
        for field in fields(data_cls):
            if not hasattr(self, field.name):
                continue
            kwargs[field.name] = _recursive_config(getattr(self, field.name))
        return data_cls(**kwargs)

    @classmethod
    def from_config(cls, cfg: Any, *args: Any, **kwargs: Any) -> Any:
        r"""Instantiates the class from a serializable configuration."""
        if getattr(cfg, '_target_', None):
            cls = _locate_cls(cfg._target_)
        elif isinstance(cfg, dict) and '_target_' in cfg:
            cls = _locate_cls(cfg['_target_'])

        data_cls = cfg.__class__
        if not is_dataclass(data_cls):
            data_cls = dataclass_from_class(cls)
            if data_cls is None:
                raise ValueError(f"Could not find the configuration class "
                                 f"that belongs to '{cls.__name__}'. Please "
                                 f"register it in the configuration store.")

        field_names = {field.name for field in fields(data_cls)}
        if isinstance(cfg, dict):
            _kwargs = {k: v for k, v in cfg.items() if k in field_names}
            cfg = data_cls(**_kwargs)
        assert is_dataclass(cfg)

        if len(args) > 0:  # Convert `*args` to `**kwargs`:
            param_names = list(inspect.signature(cls).parameters.keys())
            if 'args' in param_names:
                param_names.remove('args')
            if 'kwargs' in param_names:
                param_names.remove('kwargs')

            for name, arg in zip(param_names, args):
                kwargs[name] = arg

        for key in field_names:
            if key not in kwargs and key != '_target_':
                kwargs[key] = _recursive_from_config(getattr(cfg, key))

        return cls(**kwargs)


def _recursive_config(value: Any) -> Any:
    if isinstance(value, ConfigMixin):
        return value.config()
    if is_torch_instance(value, ConfigMixin):
        return value.config()
    if isinstance(value, (tuple, list, ModuleList)):
        return [_recursive_config(v) for v in value]
    if isinstance(value, (dict, ModuleDict)):
        return {k: _recursive_config(v) for k, v in value.items()}
    return value


def _recursive_from_config(value: Any) -> Any:
    cls: Any = None
    if is_dataclass(value):
        if getattr(value, '_target_', None):
            try:
                cls = _locate_cls(value._target_)  # type: ignore
            except ImportError:
                pass  # Keep the dataclass as it is.
        else:
            cls = class_from_dataclass(value.__class__)
    elif isinstance(value, dict) and '_target_' in value:
        cls = _locate_cls(value['_target_'])

    if cls is not None and issubclass(cls, ConfigMixin):
        return cls.from_config(value)
    if isinstance(value, (tuple, list)):
        return [_recursive_from_config(v) for v in value]
    if isinstance(value, dict):
        return {k: _recursive_from_config(v) for k, v in value.items()}
    return value


def _locate_cls(qualname: str) -> Any:
    parts = qualname.split('.')

    if len(parts) <= 1:
        raise ValueError(f"Qualified name is missing a dot (got '{qualname}')")

    if any([len(part) == 0 for part in parts]):
        raise ValueError(f"Relative imports not supported (got '{qualname}')")

    module_name, cls_name = '.'.join(parts[:-1]), parts[-1]
    return getattr(import_module(module_name), cls_name)
