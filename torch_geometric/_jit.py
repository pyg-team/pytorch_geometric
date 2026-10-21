r"""Compatibility module for :pytorch:`PyTorch` JIT functionality.

Provides version-aware wrappers for ``torch.jit`` primitives that handle the
deprecation of ``torch.jit.script`` in PyTorch >= 2.12 by falling back to
no-ops or ``typing`` equivalents.
"""

import typing
import warnings
from typing import Any, Callable, TypeVar

import torch

import torch_geometric.typing

T = TypeVar('T')

# ---------------------------------------------------------------------------
# is_scripting
# ---------------------------------------------------------------------------
if torch_geometric.typing.WITH_PT212:

    def is_scripting() -> bool:
        r"""Returns :obj:`False` since TorchScript is deprecated in
        PyTorch >= 2.12.
        """
        return False
else:
    is_scripting = torch.jit.is_scripting

# ---------------------------------------------------------------------------
# is_tracing
# ---------------------------------------------------------------------------
if torch_geometric.typing.WITH_PT212:

    def is_tracing() -> bool:
        r"""Returns :obj:`False` since TorchScript tracing is deprecated in
        PyTorch >= 2.12.
        """
        return False
else:
    is_tracing = torch.jit.is_tracing

# ---------------------------------------------------------------------------
# script
# ---------------------------------------------------------------------------
if torch_geometric.typing.WITH_PT212:

    def script(obj: T, *args: Any, **kwargs: Any) -> T:
        r"""In PyTorch >= 2.12, :func:`torch.jit.script` is deprecated.
        Returns the object unchanged and emits a deprecation warning.
        Use :func:`torch.compile` or :func:`torch.export` instead.
        """
        warnings.warn(
            "torch.jit.script() is deprecated in PyTorch >= 2.12 "
            "and is replaced by torch.compile() or torch.export(). "
            "Returning the object unchanged.",
            FutureWarning,
            stacklevel=2,
        )
        return obj
else:
    script = torch.jit.script

# ---------------------------------------------------------------------------
# export (decorator)
# ---------------------------------------------------------------------------
if torch_geometric.typing.WITH_PT212:

    def export(fn: Callable) -> Callable:
        r"""No-op decorator since TorchScript is deprecated in
        PyTorch >= 2.12.
        """
        return fn
else:
    export = torch.jit.export

# ---------------------------------------------------------------------------
# unused (decorator)
# ---------------------------------------------------------------------------
if torch_geometric.typing.WITH_PT212:

    def unused(fn: Callable) -> Callable:
        r"""No-op decorator since TorchScript is deprecated in
        PyTorch >= 2.12.
        """
        return fn
else:
    unused = torch.jit.unused

# ---------------------------------------------------------------------------
# Attribute
# ---------------------------------------------------------------------------
if torch_geometric.typing.WITH_PT212:

    def Attribute(value: T, typ: Any) -> T:  # type: ignore
        r"""Pass-through since TorchScript is deprecated in
        PyTorch >= 2.12. Returns the value unchanged.
        """
        return value
else:
    Attribute = torch.jit.Attribute

# ---------------------------------------------------------------------------
# overload / overload_method
# ---------------------------------------------------------------------------
if torch_geometric.typing.WITH_PT212:
    overload = typing.overload
    overload_method = typing.overload
else:
    overload = torch.jit._overload  # type: ignore
    overload_method = torch.jit._overload_method  # type: ignore

# ---------------------------------------------------------------------------
# ScriptModule
# ---------------------------------------------------------------------------
if torch_geometric.typing.WITH_PT212:

    class ScriptModule:  # type: ignore
        r"""Dummy class used for :obj:`isinstance` checks.
        No module will ever be an instance of this class since TorchScript is
        deprecated in PyTorch >= 2.12.
        """
else:
    from torch.jit import ScriptModule  # type: ignore # noqa: F401
