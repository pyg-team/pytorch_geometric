from typing import Dict

__experimental_options__ = {'pytorch_scatter': False}


def is_experimental_option_enabled(option: str) -> bool:
    r"""Returns :obj:`True`, if the experimental :attr:`option` is enabled.
    See :class:`~torch_geometric.experimental.set_experimental_options` for
    a list of available options."""
    return __experimental_options__.get(option, False)


def disable_all_experimental_options() -> None:
    r"""Disables all experimental options."""
    for key in __experimental_options__:
        __experimental_options__[key] = False


class set_experimental_options:
    r"""Context-manager that sets experimental options on or off.

    :class:`set_experimental_options` will enable or disable experimental
    options based on :attr:`options` argument. It can be used as a
    context-manager or as a function.

    Example:

        >>> options = {'pytorch_scatter': True}
        >>> with torch_geometric.set_experimental_options(options):
        ...     out = model(data.x, data.edge_index)

    Args:
        options (Dict[str, bool]): Dictionary of experimental options to
            enable/disable. Valid keys:

                - **pytorch_scatter**: Enables usage of
                  :meth:`torch.scatter_reduce` instead of
                  :meth:`torch_scatter.scatter`. Requires :obj:`torch>=1.12`.
    """
    def __init__(self, options: Dict[str, bool]) -> None:
        self.previous_state = __experimental_options__.copy()
        for option, value in options.items():
            if option in __experimental_options__:
                __experimental_options__[option] = value

    def __enter__(self) -> None:
        pass

    def __exit__(self, *args) -> bool:
        __experimental_options__.update(self.previous_state)
        return False
