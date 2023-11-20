from typing import Optional

import torch.nn.functional as F
from torch import Tensor
from torch.nn import LayerNorm, Linear

from torch_geometric.nn.aggr import Aggregation


class GraphMixerAggregation(Aggregation):
    r"""The aggregation from the `"Do We Really Need Complicated
    Model Architectures for Temporal Networks?"
    <https://openreview.net/forum?id=ayPPc0SyLv1>`_ paper.

    .. math::
        \mathbf{x} = asdf

    Args:
        num_tokens (int): The number of tokens.
        in_channels (int): The input channels.
        out_channels (int): The output channels.
        dropout (float, optional): Dropout probability of the token mixing
            MLPs. (default: :obj:`0.0`)
    """
    def __init__(
        self,
        num_tokens: int,  # max_num_edges per node
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dropout = dropout
        self.token_norm = LayerNorm(in_channels)
        self.token_lin1 = Linear(num_tokens, num_tokens // 2)
        self.token_lin2 = Linear(num_tokens // 2, num_tokens)
        self.channel_norm = LayerNorm(in_channels)
        self.channel_lin1 = Linear(in_channels, 4 * in_channels)
        self.channel_lin2 = Linear(4 * in_channels, in_channels)
        self.head_norm = LayerNorm(in_channels)
        self.head_lin = Linear(in_channels, out_channels)

    def reset_parameters(self):
        pass

    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:
        # Token mixing:
        h = self.token_norm(x).mT
        h = self.token_lin1(h)
        h = F.gelu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.token_lin2(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h_token = h.mT + x

        # Channel mixing:
        h = self.channel_norm(h_token)
        h = self.channel_lin1(h)
        h = F.gelu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.channel_lin2(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h_channel = h + h_token

        # Head:
        out = self.head_norm(h_channel)
        out = out.mean(dim=1)
        out = self.head_lin(out)
        return out
