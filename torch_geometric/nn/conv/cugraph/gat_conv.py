from typing import Optional

import torch
from torch import Tensor
from torch.nn import Linear, Parameter

from torch_geometric import EdgeIndex
from torch_geometric.nn.conv.cugraph import CuGraphModule
from torch_geometric.nn.conv.cugraph.base import LEGACY_MODE
from torch_geometric.nn.inits import zeros

try:
    if LEGACY_MODE:
        from pylibcugraphops.torch.autograd import mha_gat_n2n as GATConvAgg
    else:
        from pylibcugraphops.pytorch.operators import mha_gat_n2n as GATConvAgg
except ImportError:
    pass


class CuGraphGATConv(CuGraphModule):  # pragma: no cover
    r"""The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper.

    :class:`CuGraphGATConv` is an optimized version of
    :class:`~torch_geometric.nn.conv.GATConv` based on the :obj:`cugraph-ops`
    package that fuses message passing computation for accelerated execution
    and lower memory footprint.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        bias: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope

        self.lin = Linear(in_channels, heads * out_channels, bias=False)
        self.att = Parameter(torch.empty(2 * heads * out_channels))

        if bias and concat:
            self.bias = Parameter(torch.empty(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        gain = torch.nn.init.calculate_gain('relu')
        torch.nn.init.xavier_normal_(
            self.att.view(2, self.heads, self.out_channels), gain=gain)
        zeros(self.bias)

    def forward(
        self,
        x: Tensor,
        edge_index: EdgeIndex,
        max_num_neighbors: Optional[int] = None,
    ) -> Tensor:
        graph = self.get_cugraph(edge_index, max_num_neighbors)

        x = self.lin(x)

        if LEGACY_MODE:
            out = GATConvAgg(x, self.att, graph, self.heads, 'LeakyReLU',
                             self.negative_slope, False, self.concat)
        else:
            out = GATConvAgg(x, self.att, graph, self.heads, 'LeakyReLU',
                             self.negative_slope, self.concat)

        if self.bias is not None:
            out = out + self.bias

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')
