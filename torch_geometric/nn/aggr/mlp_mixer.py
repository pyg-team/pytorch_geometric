from typing import Optional

import torch.nn.functional as F
from torch import Tensor
from torch.nn import LayerNorm, Linear, Module

from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn.models.graph_mixer import _MLPMixer


class _MLPMixer(Module):
    r"""The MLP-Mixer module.

    Args:
        num_tokens (int): Number of tokens/patches in each sample.
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        dropout (float, optional): Dropout probability. (default: :obj:`0.0`)
    """
    def __init__(
        self,
        num_tokens: int,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ) -> None:
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

    def reset_parameters(self) -> None:
        self.token_norm.reset_parameters()
        self.token_lin1.reset_parameters()
        self.token_lin2.reset_parameters()
        self.channel_norm.reset_parameters()
        self.channel_lin1.reset_parameters()
        self.channel_lin2.reset_parameters()
        self.head_norm.reset_parameters()
        self.head_lin.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        r"""Forward pass.

        Args:
            x (torch.Tensor): Tensor of size
                :obj:`[*, num_tokens, in_channels]`.

        Returns:
            Tensor of size :obj:`[*, out_channels]`.
        """
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


class MLPMixerAggregation(Aggregation):
    r"""Performs MLPMixer aggregation from the GraphMixer paper.

    Args:
        num_tokens (int): Number of edges per node. If a node has less edges,
            the remaining edges are filled with zero vectors.
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        max_num_elements (int): The maximum number of elements to aggregate per
            group.
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.models.MLP`.
    """
    def __init__(
        self,
        num_tokens: int,
        in_channels: int,
        out_channels: int,
        max_num_elements: int,
        **kwargs,
    ) -> None:
        super().__init__(max_num_elements=max_num_elements)
        self.mlp_mixer = _MLPMixer(
            num_tokens=30,
            in_channels=5,
            out_channels=7,
            dropout=0.5,
        )
        self.mlp_mixer.reset_parameters()

    def reset_parameters(self) -> None:
        self.mlp_mixer.reset_parameters()

    def forward(
        self,
        x: Tensor,  # [num_edges, num_feats]
        index: Optional[Tensor] = None,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
    ) -> Tensor:
        # [num_nodes, K, num_feats] -> [num_nodes, num_feats]
        x, _ = self.to_dense_batch(x, index, ptr, dim_size, dim,
                                   max_num_elements=self.max_num_elements)
        return self.mlp_mixer()
