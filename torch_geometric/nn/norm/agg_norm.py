from typing import Any, Dict, List, Optional, Union

import torch
from torch import Tensor

from torch_geometric.nn import Aggregation
from torch_geometric.nn.resolver import aggregation_resolver


class AggSubstraction(torch.nn.Module):
    def __init__(
        self,
        aggr: Optional[Union[str, List[str], Aggregation]] = "mean",
        aggr_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.aggr_module = aggregation_resolver(aggr, **(aggr_kwargs or {}))

    def forward(self, x: Tensor, index: Optional[Tensor] = None):
        return x - self.aggr_module(x, index)
