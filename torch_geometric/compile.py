import logging
import warnings
from typing import Callable, Optional

import torch

import torch_geometric.typing


def compile(model: Optional[Callable] = None, *args, **kwargs) -> Callable:
    r"""Optimizes the given :pyg:`PyG` model/function via
    :meth:`torch.compile`.

    This function has the same signature as :meth:`torch.compile`, but it
    applies further optimization to make :pyg:`PyG` models/functions more
    compiler-friendly.

    Specifically, it

    1. temporarily disables the usage of the extension packages
       :obj:`torch_scatter`, :obj:`torch_sparse` and :obj:`pyg_lib`, and

    2. converts all instances of
       :class:`~torch_geometric.nn.conv.MessagePassing` modules into their
       jittable instances
       (see :meth:`~torch_geometric.nn.conv.MessagePassing.jittable`)

    .. note::
        Without these adjustments, :meth:`torch.compile` may currently fail to
        correctly optimize your :pyg:`PyG` model.
        We are working on fully relying on :meth:`torch.compile` for future
        releases.
    """
    if model is None:

        def fn(model: Callable) -> Callable:
            return compile(model, *args, **kwargs)

        return fn

    # Temporarily disable the usage of external extension packages:
    prev_state = {
        'WITH_PYG_LIB': torch_geometric.typing.WITH_PYG_LIB,
        'WITH_SAMPLED_OP': torch_geometric.typing.WITH_SAMPLED_OP,
        'WITH_INDEX_SORT': torch_geometric.typing.WITH_INDEX_SORT,
        'WITH_TORCH_SCATTER': torch_geometric.typing.WITH_TORCH_SCATTER,
        'WITH_TORCH_SPARSE': torch_geometric.typing.WITH_TORCH_SPARSE,
    }
    warnings.filterwarnings('ignore', ".*the 'torch-scatter' package.*")
    for key in prev_state.keys():
        setattr(torch_geometric.typing, key, False)

    # Temporarily adjust the logging level of `torch.compile`:
    prev_log_level = {
        'torch._dynamo': logging.getLogger('torch._dynamo').level,
        'torch._inductor': logging.getLogger('torch._inductor').level,
    }
    log_level = kwargs.pop('log_level', logging.WARNING)
    for key in prev_log_level.keys():
        logging.getLogger(key).setLevel(log_level)

    # Replace instances of `MessagePassing` by their jittable version:
    if isinstance(model, torch_geometric.nn.MessagePassing):
        model = model.jittable()
        # TODO Apply recursively.

    out = torch.compile(model, *args, **kwargs)

    # Restore the previous state:
    for key, value in prev_state.items():
        setattr(torch_geometric.typing, key, value)
    for key, value in prev_log_level.items():
        logging.getLogger(key).setLevel(value)

    return out
