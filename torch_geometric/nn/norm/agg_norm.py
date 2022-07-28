from typing import Any, Dict, List, Optional, Union

import torch
from torch import Tensor

from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn.resolver import aggregation_resolver


class AggSubtraction(torch.nn.Module):
    r"""Applies layer normalization by subtracting an aggregation
    over the inputs. Supports providing an aggregator from the
    :class:`~torch_geometric.nn.aggr.Aggregation` module.

    Args:
        aggr (string or list or Aggregation, optional): The aggregation scheme
            to use, *e.g.*, :obj:`"add"`, :obj:`"sum"` :obj:`"mean"`,
            :obj:`"min"`, :obj:`"max"` or :obj:`"mul"`.
            In addition, can be any
            :class:`~torch_geometric.nn.aggr.Aggregation` module (or any string
            that automatically resolves to it).
            If given as a list, will make use of multiple aggregations in which
            different outputs will get concatenated in the last dimension.
            If set to :obj:`None`, the :class:`MessagePassing` instantiation is
            expected to implement its own aggregation logic via
            :meth:`aggregate`. (default: :obj:`"mean"`)
        aggr_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective aggregation function in case it gets automatically
            resolved. (default: :obj:`None`)
    """
    def __init__(
        self,
        aggr: Optional[Union[str, List[str], Aggregation]] = "mean",
        aggr_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.aggr_module = aggregation_resolver(aggr, **(aggr_kwargs or {}))

    def forward(self, x: Tensor, index: Optional[Tensor] = None):
        agg = self.aggr_module(x, index)
        if index is not None:
            agg = agg.index_select(0, index)
        return x - agg
