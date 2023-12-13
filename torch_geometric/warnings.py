import warnings
from typing import Literal

import torch

import torch_geometric


def _is_compiling() -> bool:  # pragma: no cover
    if torch_geometric.typing.WITH_PT21:
        return torch._dynamo.is_compiling()
    return False


def warn(message: str) -> None:
    if _is_compiling():
        return

    warnings.warn(message)


def filterwarnings(
    action: Literal['default', 'error', 'ignore', 'always', 'module', 'once'],
    message: str,
) -> None:
    if _is_compiling():
        return

    warnings.filterwarnings(action, message)
