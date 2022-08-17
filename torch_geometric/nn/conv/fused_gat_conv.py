from typing import Tuple, Union, Optional

import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn.conv import GATConv
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
)

from dgNN.operators import GATConvFuse

# dgNN library is needed for FusedGATConv layer, which is equivalent to the standard GAT model mathmatically with less memory consumption and faster speed, due to kernel fusion techniques, as in paper 'Understanding GNN Computational Graph: A Coordinated Computation, IO, and Memory Perspective'(https://proceedings.mlsys.org/paper/2022/hash/9a1158154dfa42caddbd0694a4e9bdc8-Abstract.html)
# dgNN library can be installed as in repo https://github.com/dgSPARSE/dgNN

class FusedGATConv(GATConv):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = 'mean',
        bias: bool = True,
        **kwargs,
    ):
        if not all([add_self_loops is False, edge_dim is None, fill_value is 'mean']):
            raise NotImplementedError('FusedGATConv does not support add_self_loops, edge_features or non-mean fill_value')

        super(FusedGATConv, self).__init__(in_channels, out_channels, heads, concat, negative_slope, dropout, add_self_loops, edge_dim, fill_value, bias, **kwargs)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj):

        H, C = self.heads, self.out_channels

        # We first transform the input node features. If a tuple is passed, we
        # transform source and target node features via separate weights:
        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = x_dst = self.lin_src(x).view(-1, H, C)
        else:  # Tuple of source and target node features:
            x_src, x_dst = x
            assert x_src.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = self.lin_src(x_src).view(-1, H, C)
            if x_dst is not None:
                x_dst = self.lin_dst(x_dst).view(-1, H, C)

        row_ptr, col_idx, col_ptr, row_idx, permute = edge_index

        # Next, we compute node-level attention coefficients, both for source
        # and target nodes (if present):
        alpha_src = (x_src * self.att_src).sum(dim=-1)
        alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)

        if self.training:
            out = GATConvFuse(alpha_dst, alpha_src, row_ptr, col_idx, col_ptr,
                              row_idx, permute, self.negative_slope, x_src,
                              self.dropout)
        else:
            out = GATConvFuse(alpha_dst, alpha_src, row_ptr, col_idx, col_ptr,
                              row_idx, permute, self.negative_slope, x_src,
                              0.0)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        return out
