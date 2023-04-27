import functools
import inspect
from typing import List, Optional, Union

__experimental_flag__ = {'disable_dynamic_shapes': False}

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
    options."""
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


def disable_dynamic_shapes(required_args: Union[list, tuple]):

    if not required_args:
        raise ValueError('required_args list cannot be empty')

    def decorator(func):
        func_spec = inspect.getfullargspec(func)

        required_args_pos = {}

        for arg_name in required_args:
            if arg_name not in func_spec.args:
                raise ValueError(
                    f'function {func} does not take a {arg_name} argument')
            required_args_pos[arg_name] = func_spec.args.index(arg_name)

        num_args = len(func_spec.args)
        num_default_args = 0 if func_spec.defaults is None else len(
            func_spec.defaults)
        num_positional_args = num_args - num_default_args

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            dynamic_shapes_disabled = is_experimental_mode_enabled(
                "disable_dynamic_shapes")
            if dynamic_shapes_disabled:
                num_passed_args = len(args)

                def validate_param(param_name, value):
                    if value is None:
                        raise ValueError(
                            "Dynamic shapes disabled. Mandatory parameter "
                            f"`{param_name}` cannot be None.")

                for param_name in required_args:
                    value = None
                    index = required_args_pos[param_name]
                    if index < num_passed_args:
                        value = args[index]
                    elif param_name in kwargs:
                        value = kwargs[param_name]
                    elif num_default_args:
                        defaults_index = index - num_positional_args

                        if defaults_index < num_default_args:
                            value = func_spec.defaults[defaults_index]

                    validate_param(param_name, value)

            return func(*args, **kwargs)

        return wrapper

    return decorator
