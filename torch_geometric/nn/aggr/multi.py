from typing import List, Optional, Union

import torch
from torch import Tensor

from torch_geometric.nn.resolver import aggregation_resolver

from .base import Aggregation


class MultiAggregation(Aggregation):
    def __init__(self, aggrs: List[Union[Aggregation, str]]):
        super().__init__()
        assert isinstance(aggrs,
                          (list, tuple)), 'aggrs should be a list or tuple'
        assert len(aggrs) > 0, f'{len(aggrs)} aggr is passed'
        self.aggrs = list(map(aggregation_resolver, aggrs))

    def forward(self, x: Tensor, index: Optional[Tensor] = None, *,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:
        outs = []
        for aggr in self.aggrs:
            tmp = aggr(x, index=index, ptr=ptr, dim_size=dim_size, dim=dim)
            outs.append(tmp)
        out = self.combine(outs)
        return out

    def combine(self, inputs: List[Tensor]) -> Tensor:
        r"""Combines the outputs from multiple aggregations into a single
        representation."""
        return torch.cat(inputs, dim=-1) if len(inputs) > 1 else inputs[0]

    def __repr__(self) -> str:
        args = [f'  {str(aggr)}' for aggr in self.aggrs]
        return '{}([\n{}\n])'.format(self.__class__.__name__, ',\n'.join(args))
