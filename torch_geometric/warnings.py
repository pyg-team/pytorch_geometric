import warnings
from typing import Literal

import torch_geometric


def warn(message: str) -> None:
    if torch_geometric.is_compiling():
        return

    warnings.warn(message)


def filterwarnings(
    action: Literal['default', 'error', 'ignore', 'always', 'module', 'once'],
    message: str,
) -> None:
    if torch_geometric.is_compiling():
        return

    warnings.filterwarnings(action, message)
