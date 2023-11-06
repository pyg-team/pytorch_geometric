import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import LayerNorm, Linear

from torch_geometric.nn import TemporalEncoding
from torch_geometric.utils import scatter, to_dense_batch


class NodeEncoder(torch.nn.Module):
    r"""The node encoder module from the `"Do We Really Need Complicated
    Model Architectures for Temporal Networks?"
    <https://openreview.net/forum?id=ayPPc0SyLv1>`_ paper.
    :class:`NodeEncoder` captures the 1-hop temporal neighborhood information
    via mean pooling.

    .. math::
        \mathbf{x}_v^{\prime}(t_0) = \mathbf{x}_v + \textrm{mean} \left\{
        \mathbf{x}_w : w \in \mathcal{N}(v, t_0 - T, t_0) \right\}

    Args:
        time_window (int): The temporal window size :math:`T` to define the
            1-hop temporal neighborhood.
    """
    def __init__(self, time_window: int):
        super().__init__()
        self.time_window = time_window

    def reset_parameters(self):
        pass

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_time: Tensor,
        seed_time: Tensor,
    ) -> Tensor:
        r"""Forward pass.

        Args:
            x (torch.Tensor): The input node features.
            edge_index (torch.Tensor): The edge indices.
            edge_time (torch.Tensor): The timestamp attached to every edge.
            seed_time (torch.Tensor): The seed time :math:`t_0` for every
                destination node.
        """
        mask = ((edge_time <= seed_time[edge_index[1]]) &
                (edge_time > seed_time[edge_index[1]] - self.time_window))

        src, dst = edge_index[:, mask]
        mean = scatter(x[src], dst, dim=0, dim_size=x.size(0), reduce='mean')
        return x + mean

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(time_window={self.time_window})'


class _MLPMixer(torch.nn.Module):
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


def get_latest_k_edge_attr(
    k: int,
    edge_index: Tensor,
    edge_attr: Tensor,
    edge_time: Tensor,
    num_nodes: int,
    is_sorted: bool = False,
) -> Tensor:
    r"""Returns the latest :obj:`k` incoming edge attributes by
    :obj:`edge_time` for each node.
    The shape of the output tensor is :obj:`[num_nodes, k, edge_attr_dim]`.
    Nodes with fewer than :obj:`k` incoming edges are zero-padded.
    """
    _, col = edge_index

    if not is_sorted:
        perm = np.lexsort([
            -edge_time.detach().cpu().numpy(),
            col.detach().cpu().numpy(),
        ])
        perm = torch.from_numpy(perm).to(edge_index.device)
        col = col[perm]
        edge_attr = edge_attr[perm]

    return to_dense_batch(
        edge_attr,
        col,
        max_num_nodes=k,
        batch_size=num_nodes,
    )[0]


class LinkEncoder(torch.nn.Module):
    r"""The link encoder module from the `"Do We Really Need Complicated
    Model Architectures for Temporal Networks?"
    <https://openreview.net/forum?id=ayPPc0SyLv1>`_ paper.
    It is composed of two components: (1) :class:`TemporalEncoding` maps each
    edge timestamp to a :obj:`time_channels`-dimensional vector; (2) an MLP
    that groups and maps the :math:`k`-latest encoded timestamps and edge
    features to a :obj:`out_channels`-dimensional representation.

    Args:
        k (int): The number of most recent temporal links to use.
        in_channels (int): The edge feature dimensionality.
        hidden_channels (int): Size of each hidden sample.
        time_channels (int): Size of encoded timestamp.
        out_channels (int): Size of each output sample.
        is_sorted (bool, optional): If set to :obj:`True`, assumes that
            :obj:`edge_index` is sorted by column and the
            rows are sorted according to :obj:`edge_time`
            within individual neighborhoods. This avoids internal
            re-sorting of the data and can improve runtime and memory
            efficiency. (default: :obj:`False`)
        dropout (float, optional): Dropout probability of the MLP layer.
            (default: :obj:`0.0`)
    """
    def __init__(
        self,
        k: int,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        time_channels: int,
        is_sorted: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.k = k
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.time_channels = time_channels
        self.is_sorted = is_sorted
        self.dropout = dropout

        self.temporal_encoder = TemporalEncoding(time_channels)
        self.temporal_head = Linear(time_channels + in_channels,
                                    hidden_channels)

        self.mlp_mixer = _MLPMixer(  # MLP that summarizes temporal embeddings:
            num_tokens=k,
            in_channels=hidden_channels,
            out_channels=out_channels,
            dropout=dropout,
        )

    def reset_parameters(self):
        self.temporal_encoder.reset_parameters()
        self.temporal_head.reset_parameters()
        self.mlp_mixer.reset_parameters()

    def forward(
        self,
        edge_index: Tensor,
        edge_attr: Tensor,
        edge_time: Tensor,
        seed_time: Tensor,
    ) -> Tensor:
        r"""Forward pass.

        Args:
            edge_index (torch.Tensor): The edge indices.
            edge_attr (torch.Tensor): The edge features of shape
                :obj:`[num_edges, in_channels]`.
            edge_time (torch.Tensor): The time tensor of shape
                :obj:`[num_edges]`. This can be in the order of millions.
            seed_time (torch.Tensor): The seed time :math:`t_0` for every
                destination node.

        Returns:
            A node embedding tensor of shape :obj:`[num_nodes, out_channels]`.
        """
        mask = edge_time <= seed_time[edge_index[1]]

        edge_index = edge_index[:, mask]
        edge_attr = edge_attr[mask]
        edge_time = edge_time[mask]

        time_enc = self.temporal_encoder(seed_time[edge_index[1]] - edge_time)
        edge_attr = torch.cat([time_enc, edge_attr], dim=-1)
        edge_attr = self.temporal_head(edge_attr)

        edge_attr = get_latest_k_edge_attr(
            k=self.k,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_time=edge_time,
            num_nodes=seed_time.size(0),
            is_sorted=self.is_sorted,
        )

        return self.mlp_mixer(edge_attr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(k={self.k}, '
                f'in_channels={self.in_channels}, '
                f'hidden_channels={self.hidden_channels}, '
                f'out_channels={self.out_channels}, '
                f'time_channels={self.time_channels}, '
                f'dropout={self.dropout})')
