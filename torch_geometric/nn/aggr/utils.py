from typing import Optional

import torch
from torch import Tensor


class MultiheadAttentionBlock(torch.nn.Module):
    r"""The Multihead Attention Block (MAB) from the `"Set Transformer: A
    Framework for Attention-based Permutation-Invariant Neural Networks"
    <https://arxiv.org/abs/1810.00825>`_ paper

    .. math::

        \mathrm{MAB}(x, y) = \mathrm{LayerNorm}(x +
        \mathrm{Multihead}(x, y, y))

    Args:
        channels (int): Size of each input sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        norm (str, optional): If set to :obj:`False`, will not apply a final
            layer normalization. (default: :obj:`True`)
    """
    def __init__(self, channels: int, heads: int = 1, layer_norm: bool = True):
        super().__init__()

        self.channels = channels
        self.heads = heads

        self.attn = torch.nn.MultiheadAttention(
            channels,
            heads,
            batch_first=True,
        )
        self.layer_norm = torch.nn.LayerNorm(channels) if layer_norm else None

    def reset_parameters(self):
        self.attn._reset_parameters()
        if self.layer_norm is not None:
            self.layer_norm.reset_parameters()

    def forward(self, x: Tensor, y: Tensor, x_mask: Optional[Tensor] = None,
                y_mask: Optional[Tensor] = None) -> Tensor:
        """"""
        if y_mask is not None:
            y_mask = ~y_mask

        out, _ = self.attn(x, y, y, y_mask, need_weights=False)

        if x_mask is not None:
            out[~x_mask] = 0.

        out = out + x

        if self.layer_norm is not None:
            out = self.layer_norm(out)

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.channels}, '
                f'heads={self.heads}, '
                f'layer_norm={self.layer_norm is not None})')
