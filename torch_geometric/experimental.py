from typing import List, Optional, Union

__experimental_flag__ = {}

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

    Example:

        >>> with torch_geometric.experimental_mode():
        ...     out = model(data.x, data.edge_index)

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
