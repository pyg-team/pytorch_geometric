from typing import Any, Dict, List, Optional, Union

import torch
from torch import Tensor
from torch.nn import Linear
from torch_scatter import scatter

from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn.resolver import aggregation_resolver


class MultiAggregation(Aggregation):
    def __init__(self, aggrs: List[Union[Aggregation, str]],
                 aggrs_kwargs: Optional[List[Dict[str, Any]]] = None,
                 combine_mode: str = 'cat', in_channels: Optional[int] = None,
                 out_channels: Optional[int] = None):

        super().__init__()

        if not isinstance(aggrs, (list, tuple)):
            raise ValueError(f"'aggrs' of '{self.__class__.__name__}' should "
                             f"be a list or tuple (got '{type(aggrs)}').")

        if len(aggrs) == 0:
            raise ValueError(f"'aggrs' of '{self.__class__.__name__}' should "
                             f"not be empty.")

        if aggrs_kwargs is None:
            aggrs_kwargs = [{}] * len(aggrs)
        elif len(aggrs) != len(aggrs_kwargs):
            raise ValueError(f"'aggrs_kwargs' with invalid length passed to "
                             f"'{self.__class__.__name__}' "
                             f"(got '{len(aggrs_kwargs)}', "
                             f"expected '{len(aggrs)}'). Ensure that both "
                             f"'aggrs' and 'aggrs_kwargs' are consistent.")

        self.aggrs = torch.nn.ModuleList([
            aggregation_resolver(aggr, **aggr_kwargs)
            for aggr, aggr_kwargs in zip(aggrs, aggrs_kwargs)
        ])

        self.combine_mode = combine_mode
        if combine_mode == 'proj':
            if in_channels is None:
                raise ValueError(
                    f"'Combine mode '{combine_mode}' must "
                    f"have `in_channels` specified (got '{in_channels}').")
            if out_channels is None:
                out_channels = in_channels
            self.lin = Linear(in_channels * len(aggrs), out_channels,
                              bias=True)
        else:
            if (in_channels or out_channels) is not None:
                raise ValueError("Channel projection is only supported in "
                                 "the `'proj'` combine mode.")

    def reset_parameters(self):
        for aggr in self.aggrs:
            aggr.reset_parameters()
        if self.combine_mode == 'proj':
            self.lin.reset_parameters()

    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:
        outs = []
        for aggr in self.aggrs:
            outs.append(aggr(x, index, ptr, dim_size, dim))

        return self.combine(outs) if len(outs) > 1 else outs[0]

    def combine(self, inputs: List[Tensor]) -> Tensor:
        if self.combine_mode in ['cat', 'proj']:
            out = torch.cat(inputs, dim=-1)
            return self.lin(out) if hasattr(self, 'lin') else out
        elif self.combine_mode in ['sum', 'add', 'mul', 'mean', 'min', 'max']:
            return scatter(torch.stack(inputs, dim=0),
                           torch.tensor([0] * len(inputs)), dim=0,
                           reduce=self.combine_mode).squeeze_()
        else:
            raise ValueError(f"'Combine mode: '{self.combine_mode}' is not "
                             f"supported.")

    def __repr__(self) -> str:
        args = [f'  {aggr}' for aggr in self.aggrs]
        return '{}([\n{}\n], combine_mode={})'.format(
            self.__class__.__name__,
            ',\n'.join(args),
            self.combine_mode,
        )
