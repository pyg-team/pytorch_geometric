import warnings
from typing import Literal

from torch_geometric.typing import is_compiling


def warn(message: str) -> None:
    if is_compiling():
        return

    warnings.warn(message)


def filterwarnings(
    action: Literal['default', 'error', 'ignore', 'always', 'module', 'once'],
    message: str,
) -> None:
    if is_compiling():
        return

    warnings.filterwarnings(action, message)
