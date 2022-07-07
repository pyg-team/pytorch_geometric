from typing import Any, Dict, List, Optional, Union

import torch
from torch import Tensor

from torch_geometric.nn.aggr import Aggregation, MultiAggregation
from torch_geometric.utils import degree


class DegreeScalerAggregation(Aggregation):
    """
    Class that combines together one or more aggregators and then transforms
    the result with one or more scalers. The scalers are normalised by the
    in-degree of the training set and so must be provided at construction.

    Args:
        aggrs (list of string or list or Aggregation): The list of
            aggregations given as :class:`~torch_geometric.nn.aggr.Aggregation`
            (or any string that automatically resolves to it).
        scalers (list of str): Set of scaling function identifiers, namely
                :obj:`"identity"`, :obj:`"amplification"`,
                :obj:`"attenuation"`, :obj:`"linear"` and
                :obj:`"inverse_linear"`.
        deg (Tensor): Histogram of in-degrees of nodes in the training set,
            used by scalers to normalize.
        aggr_kwargs (List[Dict[str, Any]], optional): Arguments passed to the
            respective aggregation functions in case it gets automatically
            resolved. (default: :obj:`None`)
    """
    def __init__(self, aggrs: List[Union[Aggregation, str]],
                 scalers: List[str], deg: Tensor,
                 aggrs_kwargs: Optional[List[Dict[str, Any]]] = None):

        super().__init__()

        # TODO: Support non-lists
        if not isinstance(aggrs, list):
            raise RuntimeError("`aggrs` must be a list of aggregations ")

        self.aggr = MultiAggregation(aggrs, aggrs_kwargs)
        self.scalers = scalers

        deg = deg.to(torch.float)
        num_nodes = int(deg.sum())
        bin_degrees = torch.arange(deg.numel())
        self.avg_deg: Dict[str, float] = {
            'lin': float((bin_degrees * deg).sum()) / num_nodes,
            'log': float(((bin_degrees + 1).log() * deg).sum()) / num_nodes,
            'exp': float((bin_degrees.exp() * deg).sum()) / num_nodes,
        }

    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:

        self.assert_index_present(index)

        out = self.aggr(x, index, ptr, dim_size, dim)
        deg = degree(index, dtype=out.dtype).clamp_(1)

        size = [1] * len(out.size())
        size[dim] = -1
        deg = deg.view(*size)
        outs = []
        for scaler in self.scalers:
            if scaler == 'identity':
                pass
            elif scaler == 'amplification':
                out = out * (torch.log(deg + 1) / self.avg_deg['log'])
            elif scaler == 'attenuation':
                out = out * (self.avg_deg['log'] / torch.log(deg + 1))
            elif scaler == 'linear':
                out = out * (deg / self.avg_deg['lin'])
            elif scaler == 'inverse_linear':
                out = out * (self.avg_deg['lin'] / deg)
            else:
                raise ValueError(f'Unknown scaler "{scaler}".')
            outs.append(out)
        return torch.cat(outs, dim=-1) if len(outs) > 1 else outs[0]
