from typing import List, Optional, Union

import torch
from torch import Tensor

from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor,
)
from torch_geometric.utils import spmm


class SimpleConv(MessagePassing):
    r"""A simple message passing operator that performs (non-trainable)
    propagation

    .. math::
        \mathbf{x}^{\prime}_i = \bigoplus_{j \in \mathcal{N(i)}} e_{ji} \cdot
        \mathbf{x}_j

    where :math:`\bigoplus` defines a custom aggregation scheme.

    Args:
        aggr (str or [str] or Aggregation, optional): The aggregation scheme
            to use, *e.g.*, :obj:`"add"`, :obj:`"sum"` :obj:`"mean"`,
            :obj:`"min"`, :obj:`"max"` or :obj:`"mul"`.
            In addition, can be any
            :class:`~torch_geometric.nn.aggr.Aggregation` module (or any string
            that automatically resolves to it). (default: :obj:`"sum"`)
        combine_root (str, optional): Specifies whether or how to combine the
            central node representation (one of :obj:`"sum"`, :obj:`"cat"`,
            :obj:`None`). (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **inputs:**
          node features :math:`(|\mathcal{V}|, F)` or
          :math:`((|\mathcal{V_s}|, F), (|\mathcal{V_t}|, *))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`
        - **outputs:** node features :math:`(|\mathcal{V}|, F)` or
          :math:`(|\mathcal{V_t}|, F)` if bipartite
    """
    def __init__(
        self,
        aggr: Optional[Union[str, List[str], Aggregation]] = "sum",
        combine_root: Optional[str] = None,
        **kwargs,
    ):
        if combine_root not in ['sum', 'cat', None]:
            raise ValueError(f"Received invalid value for 'combine_root' "
                             f"(got '{combine_root}')")

        super().__init__(aggr, **kwargs)
        self.combine_root = combine_root

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_weight: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=size)

        x_dst = x[1]
        if x_dst is not None and self.combine_root is not None:
            if self.combine_root == 'sum':
                out = out + x_dst
            elif self.combine_root == 'cat':
                out = torch.cat([x_dst, out], dim=-1)

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        return spmm(adj_t, x[0], reduce=self.aggr)
