import numpy as np
import torch
import torch.nn.functional as F

from torch_geometric.nn.aggr.utils import MultiheadAttentionBlock
from torch_geometric.nn.encoding import TemporalEncoding


class FeatEncoder(torch.nn.Module):
    r"""Returns [raw_edge_feat | TimeEncode(edge_time_stamp)].

    Args:
        time_dim (int): size of the time embedding.
        feat_dim (int): edge feature dimension.
        out_dim (int): output dimension.
    """
    def __init__(self, time_dim: int = 16, feat_dim: int = 16,
                 out_dim: int = 16):
        super().__init__()

        self.time_dim = time_dim
        self.feat_dm = feat_dim
        self.out_dim = out_dim

        self.time_encoder = TemporalEncoding(self.time_dim)
        self.feat_encoder = torch.nn.Linear(self.time_dim + self.feat_dim,
                                            self.out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        if (self.time_dim > 0):
            self.time_encoder.reset_parameters()
        self.feat_encoder.reset_parameters()

    def forward(self, edge_feats, edge_ts):
        edge_time_feats = self.time_encoder(edge_ts)
        x = torch.cat([edge_feats, edge_time_feats], dim=1)
        return self.feat_encoder(x)


class TransformerBlock(torch.nn.Module):
    r"""out = X.T + MLP_Layernorm(X.T),     # apply token mixing
    out = out.T + MLP_Layernorm(out.T) # apply channel mixing.
    """
    def __init__(self, dims, heads, channel_expansion_factor=4, dropout=0.2,
                 use_single_layer=False):
        super().__init__()

        self.module_spec = ['token', 'channel']

        self.dims = dims
        if 'token' in self.module_spec:
            self.transformer_encoder = MultiheadAttentionBlock(
                dims, heads, layer_norm=False, dropout=dropout)
        if 'channel' in self.module_spec:
            self.channel_layernorm = torch.nn.LayerNorm(dims)
            self.channel_forward = GeluMLP(dims, channel_expansion_factor,
                                           dropout, use_single_layer)

    def reset_parameters(self):
        if 'token' in self.module_spec:
            self.transformer_encoder.reset_parameters()
        if 'channel' in self.module_spec:
            self.channel_layernorm.reset_parameters()
            self.channel_forward.reset_parameters()

    def token_mixer(self, x):
        x = self.transformer_encoder(x, x)
        return x

    def channel_mixer(self, x):
        x = self.channel_layernorm(x)
        x = self.channel_forward(x)
        return x

    def forward(self, x):
        if 'token' in self.module_spec:
            x = x + self.token_mixer(x)
        if 'channel' in self.module_spec:
            x = x + self.channel_mixer(x)
        return x


class GeluMLP(torch.nn.Module):
    r"""2-layer MLP with GeLU (fancy version of ReLU) as activation."""
    def __init__(self, dims, expansion_factor, dropout=0,
                 use_single_layer=False):
        super().__init__()

        self.dims = dims
        self.use_single_layer = use_single_layer

        self.expansion_factor = expansion_factor
        self.dropout = dropout

        if use_single_layer:
            self.linear_0 = torch.nn.Linear(dims, dims)
        else:
            self.linear_0 = torch.nn.Linear(dims, int(expansion_factor * dims))
            self.linear_1 = torch.nn.Linear(int(expansion_factor * dims), dims)

        self.reset_parameters()

    def reset_parameters(self):
        self.linear_0.reset_parameters()
        if self.use_single_layer is False:
            self.linear_1.reset_parameters()

    def forward(self, x):
        x = self.linear_0(x)
        x = F.gelu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.use_single_layer is False:
            x = self.linear_1(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class PositionalEncoding1D(torch.nn.Module):
    def __init__(self, dims):
        r""":param channels: The last dimension of the tensor
        you want to apply pos emb to.
        """
        super().__init__()
        self.org_channels = dims
        channels = int(np.ceil(dims / 2) * 2)
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
