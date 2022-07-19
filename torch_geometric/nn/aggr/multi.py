from typing import Any, Dict, List, Optional, Union

import torch
from torch import Tensor
from torch.nn import Linear

from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn.resolver import aggregation_resolver


class MultiAggregation(Aggregation):
    r"""Performs aggregations with one or more aggregators and combines and
        aggregated results.

    Args:
        aggrs (list): The list of aggregation schemes to use.
        aggrs_kwargs (list, optional): Arguments passed to the
            respective aggregation function in case it gets automatically
            resolved. (default: :obj:`None`)
        combine_mode (string or Aggregation, optional): The combine mode
            to use for combining aggregated results from multiple aggregations
            (:obj:`"cat"`, :obj:`"proj"`, :obj:`Aggregation`).
            (default: :obj:`"cat"`)
        in_channels (int, optional): Size of each input sample. Need to be
            specified when :obj:`"proj"` is used as the `combine_mode`.
            (default: :obj:`None`)
        out_channels (int, optional): Size of each output sample after
            combination. It is only used when :obj:`"proj"` is used as the
            `combine_mode`. It will be set as the same as `in_channels` if not
            specified. (default: :obj:`None`)
    """
    def __init__(
        self,
        aggrs: List[Union[Aggregation, str]],
        aggrs_kwargs: Optional[List[Dict[str, Any]]] = None,
        combine_mode: Optional[Union[str, Aggregation]] = 'cat',
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
    ):

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
            self.combine_lin = Linear(in_channels * len(aggrs), out_channels,
                                      bias=True)
        else:
            if (in_channels or out_channels) is not None:
                raise ValueError("Channel projection is only supported in "
                                 "the `'proj'` combine mode.")
            if combine_mode != "cat":
                self.combine_aggr = aggregation_resolver(combine_mode)

    def reset_parameters(self):
        for aggr in self.aggrs:
            aggr.reset_parameters()
        if self.combine_mode == 'proj':
            self.combine_lin.reset_parameters()

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
            if hasattr(self, 'combine_lin'):
                return self.combine_lin(out)
            else:
                return out
        elif hasattr(self, 'combine_aggr'):
            out = torch.cat(inputs, dim=-2)
            index = torch.arange(inputs[0].size(-2),
                                 device=out.device).tile(len(inputs))
            return self.combine_aggr(out, index=index, dim=-2)
        else:
            raise ValueError(f"'Combine mode: '{self.combine_mode}' is not "
                             f"supported")

    def __repr__(self) -> str:
        args = [f'  {aggr}' for aggr in self.aggrs]
        return '{}([\n{}\n], combine_mode={})'.format(
            self.__class__.__name__,
            ',\n'.join(args),
            self.combine_mode,
        )
