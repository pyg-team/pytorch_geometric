from typing import Optional

import torch
from torch import Tensor

from torch_geometric.nn.conv import SAGEConv
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, Optional


class DirSageConv(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 alpha: Optional[float] = 0.5, **kwargs):
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_src_to_dst = SAGEConv(in_channels, out_channels,
                                        flow="source_to_target",
                                        root_weight=False, bias=False)
        self.conv_dst_to_src = SAGEConv(in_channels, out_channels,
                                        flow="target_to_source",
                                        root_weight=False, bias=False)
        self.lin_self = Linear(in_channels, out_channels)
        self.alpha = alpha

        self.reset_parameters()

    def reset_parameters(self):
        self.conv_src_to_dst.reset_parameters()
        self.conv_dst_to_src.reset_parameters()
        self.lin_self.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:

        return (self.lin_self(x) +
                (1 - self.alpha) * self.conv_src_to_dst(x, edge_index) +
                self.alpha * self.conv_dst_to_src(x, edge_index))

    def __repr__(self) -> str:
        if hasattr(self, 'in_channels') and hasattr(self, 'out_channels'):
            return (f'{self.__class__.__name__}({self.in_channels}, '
                    f'{self.out_channels})')
        return f'{self.__class__.__name__}()'
