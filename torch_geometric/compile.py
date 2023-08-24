import logging
import warnings
from typing import Callable, Optional

import torch

import torch_geometric.typing

JIT_WARNING = ("Could not convert the 'model' into a jittable version. "
               "As such, 'torch.compile' may currently fail to correctly "
               "optimize your model. 'MessagePassing.jittable()' reported "
               "the following error: {error}")


def to_jittable(model: torch.nn.Module) -> torch.nn.Module:
    if isinstance(model, torch_geometric.nn.MessagePassing):
        try:
            model = model.jittable()
        except Exception as e:
            warnings.warn(JIT_WARNING.format(error=e))

    elif isinstance(model, torch.nn.Module):
        for name, child in model.named_children():
            if isinstance(child, torch_geometric.nn.MessagePassing):
                try:
                    setattr(model, name, child.jittable())
                except Exception as e:
                    warnings.warn(JIT_WARNING.format(error=e))
            else:
                to_jittable(child)

    return model


def compile(model: Optional[Callable] = None, *args, **kwargs) -> Callable:
    r"""Optimizes the given :pyg:`PyG` model/function via
    :meth:`torch.compile`.

    This function has the same signature as :meth:`torch.compile` (see
    `here <https://pytorch.org/docs/stable/generated/torch.compile.html>`__),
    but it applies further optimization to make :pyg:`PyG` models/functions
    more compiler-friendly.

    Specifically, it

    1. temporarily disables the usage of the extension packages
       :obj:`torch_scatter`, :obj:`torch_sparse` and :obj:`pyg_lib`

    2. converts all instances of
       :class:`~torch_geometric.nn.conv.MessagePassing` modules into their
       jittable instances
       (see :meth:`torch_geometric.nn.conv.MessagePassing.jittable`)

    .. note::
        Without these adjustments, :meth:`torch.compile` may currently fail to
        correctly optimize your :pyg:`PyG` model.
        We are working on fully relying on :meth:`torch.compile` for future
        releases.
    """
    if model is None:

        def fn(model: Callable) -> Callable:
            if model is None:
                raise RuntimeError("'model' cannot be 'None'")
            return compile(model, *args, **kwargs)

        return fn

    # Disable the usage of external extension packages:
    # TODO (matthias) Disable only temporarily
    prev_state = {
        'WITH_INDEX_SORT': torch_geometric.typing.WITH_INDEX_SORT,
        'WITH_TORCH_SCATTER': torch_geometric.typing.WITH_TORCH_SCATTER,
    }
    warnings.filterwarnings('ignore', ".*the 'torch-scatter' package.*")
    for key in prev_state.keys():
        setattr(torch_geometric.typing, key, False)

    # Adjust the logging level of `torch.compile`:
    # TODO (matthias) Disable only temporarily
    prev_log_level = {
        'torch._dynamo': logging.getLogger('torch._dynamo').level,
        'torch._inductor': logging.getLogger('torch._inductor').level,
    }
    log_level = kwargs.pop('log_level', logging.WARNING)
    for key in prev_log_level.keys():
        logging.getLogger(key).setLevel(log_level)

    # Replace instances of `MessagePassing` by their jittable version:
    model = to_jittable(model)

    # Finally, run `torch.compile` to create an optimized version:
    out = torch.compile(model, *args, **kwargs)

    return out
