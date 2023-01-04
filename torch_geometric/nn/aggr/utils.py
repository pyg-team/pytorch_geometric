from typing import Optional

import torch
from torch import Tensor
from torch.nn import LayerNorm, Linear, MultiheadAttention, Parameter


class MultiheadAttentionBlock(torch.nn.Module):
    r"""The Multihead Attention Block (MAB) from the `"Set Transformer: A
    Framework for Attention-based Permutation-Invariant Neural Networks"
    <https://arxiv.org/abs/1810.00825>`_ paper

    .. math::

        \mathrm{MAB}(\mathbf{x}, \mathbf{y}) &= \mathrm{LayerNorm}(\mathbf{h} +
        \mathbf{W} \mathbf{h})

        \mathbf{h} &= \mathrm{LayerNorm}(\mathbf{x} +
        \mathrm{Multihead}(\mathbf{x}, \mathbf{y}, \mathbf{y}))

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

        self.attn = MultiheadAttention(
            channels,
            heads,
            batch_first=True,
        )
        self.lin = Linear(channels, channels)
        self.layer_norm1 = LayerNorm(channels) if layer_norm else None
        self.layer_norm2 = LayerNorm(channels) if layer_norm else None

    def reset_parameters(self):
        self.attn._reset_parameters()
        self.lin.reset_parameters()
        if self.layer_norm1 is not None:
            self.layer_norm1.reset_parameters()
        if self.layer_norm2 is not None:
            self.layer_norm2.reset_parameters()

    def forward(self, x: Tensor, y: Tensor, x_mask: Optional[Tensor] = None,
                y_mask: Optional[Tensor] = None) -> Tensor:
        """"""
        if y_mask is not None:
            y_mask = ~y_mask

        out, _ = self.attn(x, y, y, y_mask, need_weights=False)

        if x_mask is not None:
            out[~x_mask] = 0.

        out = out + x

        if self.layer_norm1 is not None:
            out = self.layer_norm1(out)

        out = out + self.lin(out).relu()

        if self.layer_norm2 is not None:
            out = self.layer_norm2(out)

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.channels}, '
                f'heads={self.heads}, '
                f'layer_norm={self.layer_norm1 is not None})')


class SetAttentionBlock(torch.nn.Module):
    r"""The Set Attention Block (SAB) from the `"Set Transformer: A
    Framework for Attention-based Permutation-Invariant Neural Networks"
    <https://arxiv.org/abs/1810.00825>`_ paper

    .. math::

        \mathrm{SAB}(\mathbf{X}) = \mathrm{MAB}(\mathbf{x}, \mathbf{y})

    Args:
        channels (int): Size of each input sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        norm (str, optional): If set to :obj:`False`, will not apply a final
            layer normalization. (default: :obj:`True`)
    """
    def __init__(self, channels: int, heads: int = 1, layer_norm: bool = True):
        super().__init__()
        self.mab = MultiheadAttentionBlock(channels, heads, layer_norm)

    def reset_parameters(self):
        self.mab.reset_parameters()

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        return self.mab(x, x, mask, mask)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.mab.channels}, '
                f'heads={self.mab.heads}, '
                f'layer_norm={self.mab.layer_norm1 is not None})')


class InducedSetAttentionBlock(torch.nn.Module):
    r"""The Induced Set Attention Block (SAB) from the `"Set Transformer: A
    Framework for Attention-based Permutation-Invariant Neural Networks"
    <https://arxiv.org/abs/1810.00825>`_ paper

    .. math::

        \mathrm{ISAB}(\mathbf{X}) &= \mathrm{MAB}(\mathbf{x}, \mathbf{h})

        \mathbf{h} &= \mathrm{MAB}(\mathbf{I}, \mathbf{x})

    where :math:`\mathbf{I}` denotes :obj:`num_induced_points` learnable
    vectors.

    Args:
        channels (int): Size of each input sample.
        num_induced_points (int): Number of induced points.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        norm (str, optional): If set to :obj:`False`, will not apply a final
            layer normalization. (default: :obj:`True`)
    """
    def __init__(self, channels: int, num_induced_points: int, heads: int = 1,
                 layer_norm: bool = True):
        super().__init__()
        self.ind = Parameter(torch.Tensor(1, num_induced_points, channels))
        self.mab1 = MultiheadAttentionBlock(channels, heads, layer_norm)
        self.mab2 = MultiheadAttentionBlock(channels, heads, layer_norm)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.ind)
        self.mab1.reset_parameters()
        self.mab2.reset_parameters()

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        h = self.mab1(self.ind.expand(x.size(0), -1, -1), x, y_mask=mask)
        return self.mab2(x, h, x_mask=mask)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.ind.size(2)}, '
                f'num_induced_points={self.ind.size(1)}, '
                f'heads={self.mab1.heads}, '
                f'layer_norm={self.mab1.layer_norm1 is not None})')


class PoolingByMultiheadAttention(torch.nn.Module):
    r"""The Pooling by Multihead Attention (PMA) layer from the `"Set
    Transformer: A Framework for Attention-based Permutation-Invariant Neural
    Networks" <https://arxiv.org/abs/1810.00825>`_ paper

    .. math::

        \mathrm{PMA}(\mathbf{X}) = \mathrm{MAB}(\mathbf{S}, \mathbf{x})

    where :math:`\mathbf{S}` denotes :obj:`num_seed_points` learnable vectors.

    Args:
        channels (int): Size of each input sample.
        num_seed_points (int, optional): Number of seed points.
            (default: :obj:`1`)
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        norm (str, optional): If set to :obj:`False`, will not apply a final
            layer normalization. (default: :obj:`True`)
    """
    def __init__(self, channels: int, num_seed_points: int = 1, heads: int = 1,
                 layer_norm: bool = True):
        super().__init__()
        self.seed = Parameter(torch.Tensor(1, num_seed_points, channels))
        self.mab = MultiheadAttentionBlock(channels, heads, layer_norm)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.seed)
        self.mab.reset_parameters()

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        return self.mab(self.seed.expand(x.size(0), -1, -1), x, y_mask=mask)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.seed.size(2)}, '
                f'num_seed_points={self.seed.size(1)}, '
                f'heads={self.mab.heads}, '
                f'layer_norm={self.mab.layer_norm1 is not None})')
