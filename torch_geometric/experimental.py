from typing import List, Optional, Union

__experimental_flag__ = {'scatter_reduce': False}


def is_experimental_mode_enabled(
        options: Optional[Union[str, List[str]]] = None) -> bool:
    r"""Returns :obj:`True` if the experimental mode is enabled. See
    :class:`torch_geometric.experimental_mode` for a list of (optional)
    options."""
    if options is None:
        options = list(__experimental_flag__.keys())
    if isinstance(options, str):
        options = [options]
    return all([__experimental_flag__[option] for option in options])


class experimental_mode:
    r"""Context-manager that enables the experimental mode to test new but
    potentially unstable features.

    Example:

        >>> with torch_geometric.experimental_mode():
        ...     out = model(data.x, data.edge_index)

    Args:
        options (str or list, optional): Possible option(s):

            - :obj:`"torch_scatter"`: Enables the usage of
              :meth:`torch.scatter_reduce` instead of
              :meth:`torch_scatter.scatter`. Requires :obj:`torch>=1.12`.
    """
    def __init__(self, options: Optional[Union[str, List[str]]] = None):
        if options is None:
            options = list(__experimental_flag__.keys())
        if isinstance(options, str):
            options = [options]
        self.previous_state = {
            option: __experimental_flag__[option]
            for option in options
        }

    def __enter__(self) -> None:
        for option in self.previous_state.keys():
            __experimental_flag__[option] = True

    def __exit__(self, *args) -> bool:
        for option, value in self.previous_state.items():
            __experimental_flag__[option] = value
