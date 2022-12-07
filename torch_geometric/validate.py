from contextlib import contextmanager
from typing import Generator

__validation__ = {'enabled': __debug__}


def is_validation_enabled() -> bool:
    r"""Returns :obj:`True` if validation is enabled"""
    return __validation__['enabled']


def set_validation(value: bool):
    r"""Sets whether optional validation is enabled or disabled.

    The default behavior mimics Python's :obj:`assert`` statement:
    validation is on by default, but is disabled if Python is run in
    optimized mode (via :obj:`python -O`).
    Validation may be expensive, so you may want to disable it once a model
    is working.

    Args:
        value (bool): Whether to enable optional validation.
    """
    __validation__['enabled'] = value


@contextmanager
def validate(value: bool) -> Generator:
    r"""Creates a context-manager for managing whether any optional validation
    is enabled or disabled within a scope.

    Example:

        >>> with torch_geometric.validate(False):
        ...     out = model(data.x, data.edge_index)
    """
    try:
        prev = is_validation_enabled()
        set_validation(value)
        yield
    finally:
        set_validation(prev)
