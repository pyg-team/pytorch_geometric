import functools
import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import torch

# TODO (matthias) This file currently requires manual imports to let
# TorchScript work on decorated functions. Not totally sure why :(
from torch_geometric.utils import *  # noqa

__experimental_flag__: Dict[str, bool] = {
    'disable_dynamic_shapes': False,
}

Options = Optional[Union[str, List[str]]]


def get_options(options: Options) -> List[str]:
    if options is None:
        options = list(__experimental_flag__.keys())
    if isinstance(options, str):
        options = [options]
    return options


def is_experimental_mode_enabled(options: Options = None) -> bool:
    r"""Returns :obj:`True` if the experimental mode is enabled. See
    :class:`torch_geometric.experimental_mode` for a list of (optional)
    options.
    """
    if torch.jit.is_scripting() or torch.jit.is_tracing():
        return False
    options = get_options(options)
    return all([__experimental_flag__[option] for option in options])


def set_experimental_mode_enabled(mode: bool, options: Options = None):
    for option in get_options(options):
        __experimental_flag__[option] = mode


class experimental_mode:
    r"""Context-manager that enables the experimental mode to test new but
    potentially unstable features.

    .. code-block:: python

        with torch_geometric.experimental_mode():
            out = model(data.x, data.edge_index)

    Args:
        options (str or list, optional): Currently there are no experimental
            features.
    """
    def __init__(self, options: Options = None):
        self.options = get_options(options)
        self.previous_state = {
            option: __experimental_flag__[option]
            for option in self.options
        }

    def __enter__(self):
        set_experimental_mode_enabled(True, self.options)

    def __exit__(self, *args) -> bool:
        for option, value in self.previous_state.items():
            __experimental_flag__[option] = value


class set_experimental_mode:
    r"""Context-manager that sets the experimental mode on or off.

    :class:`set_experimental_mode` will enable or disable the experimental mode
    based on its argument :attr:`mode`.
    It can be used as a context-manager or as a function.

    See :class:`experimental_mode` above for more details.
    """
    def __init__(self, mode: bool, options: Options = None):
        self.options = get_options(options)
        self.previous_state = {
            option: __experimental_flag__[option]
            for option in self.options
        }
        set_experimental_mode_enabled(mode, self.options)

    def __enter__(self):
        pass

    def __exit__(self, *args):
        for option, value in self.previous_state.items():
            __experimental_flag__[option] = value


def disable_dynamic_shapes(required_args: List[str]) -> Callable:
    r"""A decorator that disables the usage of dynamic shapes for the given
    arguments, i.e., it will raise an error in case :obj:`required_args` are
    not passed and needs to be automatically inferred.
    """
    def decorator(func: Callable) -> Callable:
        spec = inspect.getfullargspec(func)

        required_args_pos: Dict[str, int] = {}
        for arg_name in required_args:
            if arg_name not in spec.args:
                raise ValueError(f"The function '{func}' does not have a "
                                 f"'{arg_name}' argument")
            required_args_pos[arg_name] = spec.args.index(arg_name)

        num_args = len(spec.args)
        num_default_args = 0 if spec.defaults is None else len(spec.defaults)
        num_positional_args = num_args - num_default_args

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not is_experimental_mode_enabled('disable_dynamic_shapes'):
                return func(*args, **kwargs)

            for required_arg in required_args:
                index = required_args_pos[required_arg]

                value: Optional[Any] = None
                if index < len(args):
                    value = args[index]
                elif required_arg in kwargs:
                    value = kwargs[required_arg]
                elif num_default_args > 0:
                    value = spec.defaults[index - num_positional_args]

                if value is None:
                    raise ValueError(f"Dynamic shapes disabled. Argument "
                                     f"'{required_arg}' needs to be set")

            return func(*args, **kwargs)

        return wrapper

    return decorator
