from typing import Dict, List, Optional, Union

import torch
from torch import Tensor

from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn.resolver import aggregation_resolver


class MultiAggregation(Aggregation):
    def __init__(self, aggrs: List[Union[Aggregation, str]],
                 aggrs_kwargs: Optional[List[Dict]] = None):
        super().__init__()

        if not isinstance(aggrs, (list, tuple)):
            raise ValueError(f"'aggrs' of '{self.__class__.__name__}' should "
                             f"be a list or tuple (got '{type(aggrs)}')")

        if len(aggrs) == 0:
            raise ValueError(f"'aggrs' of '{self.__class__.__name__}' should "
                             f"not be empty")

        if not aggrs_kwargs:
            aggrs_kwargs = [{}] * len(aggrs)

        self.aggrs = [
            aggregation_resolver(aggr, **arg_kwargs)
            for aggr, arg_kwargs in zip(aggrs, aggrs_kwargs)
        ]

    def forward(self, x: Tensor, index: Optional[Tensor] = None, *,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:
        outs = []
        for aggr in self.aggrs:
            outs.append(aggr(x, index, ptr=ptr, dim_size=dim_size, dim=dim))
        return torch.cat(outs, dim=-1) if len(outs) > 1 else outs[0]

    def __repr__(self) -> str:
        args = [f'  {aggr}' for aggr in self.aggrs]
        return '{}([\n{}\n])'.format(self.__class__.__name__, ',\n'.join(args))
