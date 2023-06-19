from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn import TemporalEncoding
from torch_geometric.utils import scatter, to_dense_batch
from torch_geometric.utils.num_nodes import maybe_num_nodes


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

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_time: Tensor,
        seed_time: Tensor,
    ) -> Tensor:
        r"""
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


# TODO: Generalize the module when needed
class _MLPMixer(torch.nn.Module):
    """1-layer MLP-mixer for GraphMixer.

    Args:
        num_tokens (int): The number of tokens (patches) in each sample.
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        dropout (float, optional): The dropout probability. (default: :obj:`0`)
    """
    def __init__(self, num_tokens: int, in_channels: int, out_channels: int,
                 dropout: float = 0):
        super().__init__()
        self.num_tokens = num_tokens
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout

        # token mixing
        self.token_layer_norm = torch.nn.LayerNorm((in_channels, ))
        self.token_lin_1 = torch.nn.Linear(num_tokens, num_tokens // 2)
        self.token_lin_2 = torch.nn.Linear(num_tokens // 2, num_tokens)

        # channel mixing
        self.channel_layer_norm = torch.nn.LayerNorm((in_channels, ))
        self.channel_lin_1 = torch.nn.Linear(in_channels, 4 * in_channels)
        self.channel_lin_2 = torch.nn.Linear(4 * in_channels, in_channels)

        # head
        self.head_layer_norm = torch.nn.LayerNorm((in_channels, ))
        self.head_lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (torch.Tensor): Features tensor of size
                :obj:`[N, num_tokens, in_channels]`.

        Returns:
            Tensor of size :obj:`[N, out_channels]`.
        """
        # token mixing
        h = self.token_layer_norm(x).mT
        h = self.token_lin_1(h)
        h = F.gelu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.token_lin_2(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h_token = h.mT + x

        # channel mixing
        h = self.channel_layer_norm(h_token)
        h = self.channel_lin_1(h)
        h = F.gelu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.channel_lin_2(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h_channel = h + h_token

        # head
        h_channel = self.head_layer_norm(h_channel)
        t = torch.mean(h_channel, dim=1)
        return self.head_lin(t)

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"num_tokens={self.num_tokens}, "
                f"in_channels={self.in_channels}, "
                f"out_channels={self.out_channels}, "
                f"dropout={self.dropout})")


def get_latest_k_edge_attrs(K: int, edge_index: Tensor, edge_time: Tensor,
                            edge_attr: Tensor, num_nodes: int) -> Tensor:
    r"""Returns the latest :obj:`K` incoming edge attributes by
    :obj:`edge_time` for each node. The shape
    of the output tensor is :obj:`[num_nodes, K, edge_attr_dim]`.
    Nodes with fewer than :obj:`K` incoming edges are zero-padded.
    Args:
        K (int): The number of edges to keep for each node.
        edge_index (LongTensor): The edge indices.
        edge_time (Tensor): The edge timestamps.
        edge_attr (Tensor): The edge attributes.
        num_nodes (int): The number of nodes in the graph.
    :rtype: :class:`Tensor`
    """
    assert (edge_time >= 0).all()
    _, col = edge_index
    perm = np.lexsort(
        [-edge_time.detach().cpu().numpy(),
         col.detach().cpu().numpy()])
    perm = torch.from_numpy(perm).to(edge_index.device)
    col = col[perm]
    edge_attr = edge_attr[perm]

    # zero-pad each node's edges:
    # [num_edges, hidden_channels] -> [num_nodes*K, hidden_channels]
    edge_attr, _ = to_dense_batch(
        edge_attr,
        col,
        max_num_nodes=K,
        batch_size=num_nodes,
    )
    return edge_attr


class LinkEncoder(torch.nn.Module):
    r"""The link-encoding function from the `"Do We Really Need Complicated
    Model Architectures for Temporal Networks?"
    <https://openreview.net/forum?id=ayPPc0SyLv1>`_ paper.
    It is composed of two components. The first component is
    :class:`TemporalEncoding` that maps each edge timestamp to a
    :obj:`time_channels` dimensional vector.
    The second component a 1-layer MLP that maps each encoded timestamp
    feature concatenated with its corresponding link feature to a
    :obj:`out_channels` dimensional vector.

    Args:
        K (int): The number of most recent teomporal links to use to construct
            an intermediate feature representation for each node.
        in_channels (int): Edge feature dimensionality.
        hidden_channels (int): Size of each hidden sample.
        time_channels (int): Size of encoded timestamp using
            :class:`TemporalEncoding`.
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
        K: int,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        time_channels: int,
        is_sorted: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.K = K
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.time_channels = time_channels
        self.is_sorted = is_sorted
        self.dropout = dropout

        # teomporal encoder
        self.temporal_encoder = TemporalEncoding(time_channels)
        self.temporal_encoder_head = torch.nn.Linear(
            time_channels + in_channels,
            hidden_channels,
        )

        # MLP that summarises temporal embedding.
        self.mlp_mixer = _MLPMixer(
            num_tokens=K,
            in_channels=hidden_channels,
            out_channels=out_channels,
            dropout=dropout,
        )

    def forward(
        self,
        edge_attr: Tensor,
        edge_time: Tensor,
        edge_index: Tensor,
        num_nodes: Optional[int] = None,
    ) -> Tensor:
        """
        Args:
            edge_attr (torch.Tensor): The edge features of shape
                :obj:`[num_edges, in_channels]`.
            edge_time (torch.Tensor): The time tensor of shape
                :obj:`[num_edges]`. This can be in the order of millions.
            edge_index (torch.Tensor): The edge indicies.
            num_nodes (int, optional): The number of nodes in the graph.
                (default: :obj:`None`)

        Returns:
            A node embedding tensor of shape :obj:`[num_nodes, out_channels]`.
        """
        num_nodes = maybe_num_nodes(edge_index, num_nodes)
        time_info = self.temporal_encoder(edge_time)
        edge_attr_time = torch.cat((time_info, edge_attr), dim=1)
        edge_attr_time = self.temporal_encoder_head(edge_attr_time)

        if not self.is_sorted:
            edge_attr_time = get_latest_k_edge_attrs(self.K, edge_index,
                                                     edge_time, edge_attr_time,
                                                     num_nodes)

        return self.mlp_mixer(
            edge_attr_time.view(-1, self.K, self.hidden_channels))

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"K={self.K}, "
                f"in_channels={self.in_channels}, "
                f"hidden_channels={self.hidden_channels}, "
                f"out_channels={self.out_channels}, "
                f"time_channels={self.time_channels}, "
                f"dropout={self.dropout})")
