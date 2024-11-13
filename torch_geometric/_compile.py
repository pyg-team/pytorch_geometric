import warnings
from typing import Any, Callable, Optional, Union

import torch

import torch_geometric.typing


def is_compiling() -> bool:
    r"""Returns :obj:`True` in case :pytorch:`PyTorch` is compiling via
    :meth:`torch.compile`.
    """
    if torch_geometric.typing.WITH_PT23:
        return torch.compiler.is_compiling()
    if torch_geometric.typing.WITH_PT21:
        return torch._dynamo.is_compiling()
    return False  # pragma: no cover


def compile(
    model: Optional[torch.nn.Module] = None,
    *args: Any,
    **kwargs: Any,
) -> Union[torch.nn.Module, Callable[[torch.nn.Module], torch.nn.Module]]:
    r"""Optimizes the given :pyg:`PyG` model/function via
    :meth:`torch.compile`.
    This function has the same signature as :meth:`torch.compile` (see
    `here <https://pytorch.org/docs/stable/generated/torch.compile.html>`__).

    Args:
        model: The model to compile.
        *args: Additional arguments of :meth:`torch.compile`.
        **kwargs: Additional keyword arguments of :meth:`torch.compile`.

    .. note::
        :meth:`torch_geometric.compile` is deprecated in favor of
        :meth:`torch.compile`.
    """
    warnings.warn("'torch_geometric.compile' is deprecated in favor of "
                  "'torch.compile'")
    return torch.compile(model, *args, **kwargs)  # type: ignore
