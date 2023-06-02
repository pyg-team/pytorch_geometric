import math

import torch
from torch import Tensor


class PositionalEncoding(torch.nn.Module):
    r"""The positional encoding scheme from the `"Attention Is All You Need"
    <https://arxiv.org/pdf/1706.03762.pdf>`_ paper

    .. math::

        PE(x)_{2 \cdot i} &= \sin(x / 10000^{2 \cdot i / d})

        PE(x)_{2 \cdot i + 1} &= \cos(x / 10000^{2 \cdot i / d})

    where :math:`x` is the position and :math:`i` is the dimension.

    Args:
        out_channels (int): Size :math:`d` of each output sample.
        base_freq (float, optional): The base frequency of sinusoidal
            functions. (default: :obj:`1e-4`)
        granularity (float, optional): The granularity of the positions. If
            set to smaller value, the encoder will capture more fine-grained
            changes in positions. (default: :obj:`1.0`)
    """
    def __init__(
        self,
        out_channels: int,
        base_freq: float = 1e-4,
        granularity: float = 1.0,
    ):
        super().__init__()

        if out_channels % 2 != 0:
            raise ValueError(f"Cannot use sinusoidal positional encoding with "
                             f"odd 'out_channels' (got {out_channels}).")

        self.out_channels = out_channels
        self.base_freq = base_freq
        self.granularity = granularity

        frequency = torch.logspace(0, 1, out_channels // 2, base_freq)
        self.register_buffer('frequency', frequency)

        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, x: Tensor) -> Tensor:
        """"""
        x = x / self.granularity if self.granularity != 1.0 else x
        out = x.view(-1, 1) * self.frequency.view(1, -1)
        return torch.cat([torch.sin(out), torch.cos(out)], dim=-1)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.out_channels})'


class TemporalEncoding(torch.nn.Module):
    r"""The time-encoding function from the `"Do We Really Need Complicated
    Model Architectures for Temporal Networks?"
    <https://openreview.net/forum?id=ayPPc0SyLv1>`_ paper.
    :class:`TemporalEncoding` first maps each entry to a vector with
    monotonically exponentially decreasing values, and then uses the cosine
    function to project all values to range :math:`[-1, 1]`

    .. math::
        y_{i} = \cos \left(x \cdot \sqrt{d}^{-(i - 1)/\sqrt{d}} \right)

    where :math:`d` defines the output feature dimension, and
    :math:`1 \leq i \leq d`.

    Args:
        out_channels (int): Size :math:`d` of each output sample.
    """
    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = out_channels

        sqrt = math.sqrt(out_channels)
        weight = 1.0 / sqrt**torch.linspace(0, sqrt, out_channels).view(1, -1)
        self.register_buffer('weight', weight)

        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, x: Tensor) -> Tensor:
        """"""
        return torch.cos(x.view(-1, 1) @ self.weight)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.out_channels})'


# TODO: Generalize the module when needed
class _MLPMixer(torch.nn.Module):
    """1-layer MLP-mixer for GraphMixer.

    Args:
        num_tokens (int): The number of tokens (patches) in each sample.
        in_channels (int): Input channels.
        out_channels (int): Output channels.
    """
    def __init__(
        self,
        num_tokens: int,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        # token mixing
        self.token_layer_norm = torch.nn.LayerNorm((in_channels, ))
        self.token_lin_1 = torch.nn.Linear(num_tokens, num_tokens)
        self.token_lin_2 = torch.nn.Linear(num_tokens, num_tokens)

        # channel mixing
        self.channel_layer_norm = torch.nn.LayerNorm((in_channels, ))
        self.channel_lin_1 = torch.nn.Linear(in_channels, in_channels)
        self.channel_lin_2 = torch.nn.Linear(in_channels, in_channels)

        # head
        self.head_layer_norm = torch.nn.LayerNorm((in_channels, ))
        self.head_lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # token mixing
        h = self.token_layer_norm(x).mT
        h = self.token_lin_1(h)
        h = torch.nn.functional.gelu(h)
        h = torch.nn.functional.dropout(h, training=self.training)
        h = self.token_lin_2(h)
        h = torch.nn.functional.dropout(h, training=self.training)
        h_token = h.mT + x

        # channel mixing
        h = self.channel_layer_norm(h_token)
        h = self.channel_lin_1(h)
        h = torch.nn.functional.gelu(h)
        h = torch.nn.functional.dropout(h, training=self.training)
        h = self.channel_lin_2(h)
        h = torch.nn.functional.dropout(h, training=self.training)
        h_channel = h + h_token

        # head
        h_channel = self.head_layer_norm(h_channel)
        t = torch.mean(h_channel, dim=1)
        return self.head_lin(t)

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"num_tokens={self.num_tokens}, "
                f"in_channels={self.in_channels}, "
                f"out_channels={self.out_channels})")


class LinkEncoding(torch.nn.Module):
    r"""The link-encoding function from the `"Do We Really Need Complicated
    Model Architectures for Temporal Networks?"
    <https://openreview.net/forum?id=ayPPc0SyLv1>`_ paper.
    :class:`LinkEncoding` is composed of two components. The first component is
    :class:`TemporalEncoding` that maps each edge timestamp to a
    ``time_channels`` dimensional vector.
    The second component, 1-layer MLP-mixer, maps the encoded timestamp feature
    concatenated with its corresponding link feature to a ``out_channels``
    dimensional vector.

    Args:
        K (int): The number of most recent teomporal links to use to construct
            an intermediate feature representation for each node. If the number
            of teomporal links that their node has is less than :math:`K`, the
            intermediate feature matrix will be zero-padded.
        num_edge_features (int): The number of edge features used to construct
            a linear layer encoding the output of :class:`TemporalEncoding`
            concatenated with ``edge_attr``.
        time_channels (int): dims to encode each timestamp into with
            :class:`TemporalEncoding`.
        out_channels (int): Size of each output sample.

    Example:

        >>> # GraphMixer paper uses the following args for GDELTLite dataset
        >>> link_encoder = LinkEncoding(
        ...     K=30,
        ...     num_edge_features=186,
        ...     hidden_channels=100,
        ...     out_channels=100,
        ...     time_channels=100,
        ... )

    """
    def __init__(
        self,
        K: int,
        num_edge_features: int,
        hidden_channels: int,
        out_channels: int,
        time_channels: int,
    ) -> None:
        super().__init__()
        self.K = K
        self.num_edge_features = num_edge_features
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.time_channels = time_channels

        # teomporal encoder
        self.temporal_encoder = TemporalEncoding(time_channels)
        self.temporal_encoder_head = torch.nn.Linear(
            time_channels + num_edge_features,
            hidden_channels,
        )

        # temporal information summariser
        self.mlp_mixer = _MLPMixer(
            num_tokens=K,
            in_channels=hidden_channels,
            out_channels=out_channels,
        )

    def forward(
        self,
        edge_attr: torch.Tensor,
        edge_time: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            edge_attr (torch.Tensor): ``[N, num_edge_features]``
            edge_time (torch.Tensor): ``[N,]``

        Returns:
            A tensor of size ``[N, out_channels]``.
        """
        time_info = self.temporal_encoder(edge_time)
        edge_attr_with_time = torch.cat((time_info, edge_attr), dim=1)
        edge_attr_with_time = self.temporal_encoder_head(edge_attr_with_time)
        return self.mlp_mixer(edge_attr_with_time)

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"K={self.K}, "
                f"num_edge_features={self.num_edge_features}, "
                f"hidden_channels={self.hidden_channels}, "
                f"out_channels={self.out_channels}, "
                f"time_channels={self.time_channels})")
