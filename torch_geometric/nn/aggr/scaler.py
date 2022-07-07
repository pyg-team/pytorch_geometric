from typing import Any, Dict, List, Optional, Union

import torch
from torch import Tensor

from torch_geometric.nn.aggr import Aggregation, MultiAggregation
from torch_geometric.nn.resolver import aggregation_resolver as aggr_resolver
from torch_geometric.utils import degree


class DegreeScalerAggregation(Aggregation):
    r"""Combines one or more aggregators and transforms its output with one or
    more scalers as introduced in the `"Principal Neighbourhood Aggregation for
    Graph Nets" <https://arxiv.org/abs/2004.05718>`_ paper.
    The scalers are normalised by the in-degree of the training set and so must
    be provided at time of construction.
    See :class:`torch_geometric.nn.conv.PNAConv` for more information.

    Args:
        aggr (string or list or Aggregation): The aggregation scheme to use.
            See :class:`~torch_geometric.nn.conv.MessagePassing` for more
            information.
        scaler (str or list): Set of scaling function identifiers, namely one
            or more of :obj:`"identity"`, :obj:`"amplification"`,
            :obj:`"attenuation"`, :obj:`"linear"` and :obj:`"inverse_linear"`.
        deg (Tensor): Histogram of in-degrees of nodes in the training set,
            used by scalers to normalize.
        aggr_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective aggregation function in case it gets automatically
            resolved. (default: :obj:`None`)
    """
    def __init__(
        self,
        aggr: Union[str, List[str], Aggregation],
        scaler: Union[str, List[str]],
        deg: Tensor,
        aggr_kwargs: Optional[List[Dict[str, Any]]] = None,
    ):
        super().__init__()

        if isinstance(aggr, (str, Aggregation)):
            self.aggr = aggr_resolver(aggr, **(aggr_kwargs or {}))
        elif isinstance(aggr, (tuple, list)):
            self.aggr = MultiAggregation(aggr, aggr_kwargs)
        else:
            raise ValueError(f"Only strings, list, tuples and instances of"
                             f"`torch_geometric.nn.aggr.Aggregation` are "
                             f"valid aggregation schemes (got '{type(aggr)}')")

        self.scaler = [scaler] if isinstance(aggr, str) else scaler

        deg = deg.to(torch.float)
        num_nodes = int(deg.sum())
        bin_degrees = torch.arange(deg.numel(), device=deg.device)
        self.avg_deg: Dict[str, float] = {
            'lin': float((bin_degrees * deg).sum()) / num_nodes,
            'log': float(((bin_degrees + 1).log() * deg).sum()) / num_nodes,
            'exp': float((bin_degrees.exp() * deg).sum()) / num_nodes,
        }

    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:

        # TODO Currently, `degree` can only operate on `index`:
        self.assert_index_present(index)

        out = self.aggr(x, index, ptr, dim_size, dim)

        assert index is not None
        deg = degree(index, dtype=out.dtype).clamp_(1)
        size = [1] * len(out.size())
        size[dim] = -1
        deg = deg.view(size)

        outs = []
        for scaler in self.scaler:
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
                raise ValueError(f"Unknown scaler '{scaler}'")
            outs.append(out)

        return torch.cat(outs, dim=-1) if len(outs) > 1 else outs[0]
