import warnings

import torch

import torch_geometric


def _is_compiling() -> bool:  # pragma: no cover
    if torch_geometric.typing.WITH_PT21:
        return torch._dynamo.is_compiling()
    return False


def warn(message: str):
    if _is_compiling():
        return

    warnings.warn(message)


def filterwarnings(action: str, message: str):
    if _is_compiling():
        return

    warnings.filterwarnings(action, message)
