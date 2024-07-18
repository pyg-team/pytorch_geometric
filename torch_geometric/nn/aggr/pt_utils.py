from typing import Optional

import numpy as np
import torch
from torch import Tensor

from torch_geometric.nn.aggr.utils import MultiheadAttentionBlock
from torch_geometric.nn.encoding import TemporalEncoding


class FeatEncoder(torch.nn.Module):
    r"""Returns embedded edge features.

    Args:
        feat_dim (int): edge feature dimension.
        out_dim (int): output dimension.
        time_dim (int, optional): time embedding dimension.
    """
    def __init__(
        self,
        feat_dim: int = 16,
        out_dim: int = 16,
        time_dim: int = 0,
    ):
        super().__init__()

        self.feat_dim = feat_dim
        self.out_dim = out_dim
        self.time_dim = time_dim
        if self.time_dim > 0:
            self.time_encoder = TemporalEncoding(self.time_dim)
        self.feat_encoder = torch.nn.Linear(self.time_dim + self.feat_dim,
                                            self.out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        if (self.time_dim > 0):
            self.time_encoder.reset_parameters()
        self.feat_encoder.reset_parameters()

    def forward(self, edge_feats: Tensor, edge_ts: Tensor = None):
        if (self.time_dim > 0):
            edge_time_feats = self.time_encoder(edge_ts)
            x = torch.cat([edge_feats, edge_time_feats], dim=1)
        else:
            x = edge_feats
        return self.feat_encoder(x)


class TransformerBlock(torch.nn.Module):
    r"""transformer block for patch transformer.

    Args:
        hidden_channels (int): Hidden dimension.
        heads (int): Number of attention heads.
        dropout (float, optional): dropout value.
    """
    def __init__(self, hidden_channels: int, heads: int, dropout: float = 0.2):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.transformer_encoder = MultiheadAttentionBlock(
            hidden_channels, heads, layer_norm=True, dropout=dropout)

    def reset_parameters(self):
        self.transformer_encoder.reset_parameters()

    def forward(self, x, mask: Optional[Tensor] = None):
        x = x + self.transformer_encoder(x, x, mask, mask)
        return x


class PositionalEncoding1D(torch.nn.Module):
    def __init__(self, hidden_channels):
        r"""1D positional encoding.

        Args:
            hidden_channels (int): hidden dimension.
        """
        super().__init__()
        self.org_channels = hidden_channels
        channels = int(np.ceil(hidden_channels / 2) * 2)
        self.channels = channels
        channel_range = torch.arange(0, channels, 2).float() / channels
        inv_freq = 1.0 / (1000**channel_range)
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    def get_emb(self, sin_inp):
        r"""Gets a base embedding for one dimension,
        with sin and cos intertwined.
        """
        emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
        return torch.flatten(emb, -2, -1)

    def forward(self, tensor):
        r"""Param tensor: A 3d tensor of size (batch_size, x, ch),
        return: Positional Encoding Matrix of size (batch_size, x, ch).
        """
        if len(tensor.shape) != 3:
            raise RuntimeError("need to be 3d!")

        if self.cached_penc is not None:
            if self.cached_penc.shape == tensor.shape:
                return self.cached_penc

        self.cached_penc = None
        batch_size, x, orig_ch = tensor.shape
        pos_x = torch.arange(x,
                             device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = self.get_emb(sin_inp_x)
        emb = torch.zeros((x, self.channels),
                          device=tensor.device).type(tensor.type())
        emb[:, :self.channels] = emb_x
        self.cached_penc = emb[None, :, :orig_ch].repeat(batch_size, 1, 1)
        return self.cached_penc
