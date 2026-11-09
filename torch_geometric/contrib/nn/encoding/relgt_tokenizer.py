from typing import Any, Dict, List, Optional

import torch
from torch import Tensor
from torch.nn import Embedding

from torch_geometric.contrib.nn.encoding.subgraph_gnn_pe import SubgraphGNNPE
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.encoding import PositionalEncoding

SECONDS_PER_DAY = 60 * 60 * 24


class RelGTTokenizer(torch.nn.Module):
    r"""The multi-element tokenizer from the `"Relational Graph Transformer"
    <https://arxiv.org/abs/2505.10960>`_ paper.

    Given :math:`K` sampled nodes :math:`v_j` for each of :math:`B` seed nodes
    :math:`v_i`, every sampled node is turned into a token by embedding five
    elements and mixing their concatenation:

    .. math::
        \mathbf{h}_{\textrm{token}}(v_j) = \mathbf{O} \left[
        \mathbf{h}_{\textrm{feat}}(v_j) \, \Vert \,
        \mathbf{h}_{\textrm{type}}(v_j) \, \Vert \,
        \mathbf{h}_{\textrm{hop}}(v_i, v_j) \, \Vert \,
        \mathbf{h}_{\textrm{time}}(v_i, v_j) \, \Vert \,
        \mathbf{h}_{\textrm{PE}}(v_j) \right]

    with :math:`\mathbf{O} \in \mathbb{R}^{d \times 5d}`, where

    * :math:`\mathbf{h}_{\textrm{feat}}` encodes the row features of
      :math:`v_j` via the given :obj:`feature_encoder`,
    * :math:`\mathbf{h}_{\textrm{type}}` embeds the node type (*i.e.*, table)
      of :math:`v_j`,
    * :math:`\mathbf{h}_{\textrm{hop}}` embeds the hop distance between
      :math:`v_i` and :math:`v_j`,
    * :math:`\mathbf{h}_{\textrm{time}}` encodes the relative time
      :math:`\tau(v_i) - \tau(v_j)` (converted from seconds to days) via
      :class:`~torch_geometric.nn.encoding.PositionalEncoding` followed by a
      linear projection, and
    * :math:`\mathbf{h}_{\textrm{PE}}` is a positional encoding computed on
      the sampled subgraphs (by default,
      :class:`~torch_geometric.contrib.nn.encoding.SubgraphGNNPE`).

    Hop distances :math:`0, \ldots, \textrm{max\_hops}` denote sampled
    neighbors, while the extra hop value :obj:`max_hops + 1` denotes randomly
    sampled filler nodes used when fewer than :math:`K` neighbors exist.

    Args:
        channels (int): Size :math:`d` of each token. Needs to be even.
        node_types (List[str]): The node types (*i.e.*, table names). The
            :obj:`i`-th entry is the node type with ID :obj:`i` in the
            :obj:`node_type` input of :meth:`forward`.
        feature_encoder (torch.nn.Module): A module mapping a dictionary of
            per-node-type inputs (*e.g.*, :class:`torch_frame.TensorFrame`
            objects) to a dictionary of node features of shape
            :obj:`[num_nodes_of_type, channels]`, such as the
            :obj:`HeteroEncoder` of the RelBench pipeline.
        max_hops (int, optional): The maximum hop distance of sampled
            neighbors. (default: :obj:`2`)
        pe_encoder (torch.nn.Module, optional): The positional encoder,
            called as :obj:`pe_encoder(edge_index, num_nodes, batch=batch,
            batch_size=batch_size)` and returning a tensor of shape
            :obj:`[num_nodes, channels]`.
            If set to :obj:`None`, will use
            :class:`~torch_geometric.contrib.nn.encoding.SubgraphGNNPE`
            with :obj:`channels` output channels. (default: :obj:`None`)

    Shapes:
        - **input:**
          per-node-type inputs to :obj:`feature_encoder`,
          node indices :math:`(B, K)`,
          node type IDs :math:`(B, K)`,
          hop distances :math:`(B, K)`,
          relative times in seconds :math:`(B, K)`,
          subgraph edge indices :math:`(2, |\mathcal{E}|)`
        - **output:** tokens :math:`(B, K, d)`
    """
    def __init__(
        self,
        channels: int,
        node_types: List[str],
        feature_encoder: torch.nn.Module,
        max_hops: int = 2,
        pe_encoder: Optional[torch.nn.Module] = None,
    ):
        super().__init__()

        if len(node_types) == 0:
            raise ValueError("'node_types' needs to hold at least one entry")
        if len(set(node_types)) != len(node_types):
            raise ValueError(f"'node_types' must not hold duplicate entries "
                             f"(got {node_types})")
        if max_hops < 0:
            raise ValueError(f"'max_hops' must be non-negative "
                             f"(got {max_hops})")

        self.channels = channels
        self.node_types = list(node_types)
        self.max_hops = max_hops

        self.feature_encoder = feature_encoder
        self.type_emb = Embedding(len(node_types), channels)
        # Hops `0, ..., max_hops` plus one extra entry for filler nodes:
        self.hop_emb = Embedding(max_hops + 2, channels)
        self.time_enc = PositionalEncoding(channels)
        self.time_lin = Linear(channels, channels)
        if pe_encoder is None:
            pe_encoder = SubgraphGNNPE(channels)
        self.pe_encoder = pe_encoder
        self.lin = Linear(5 * channels, channels)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        if hasattr(self.feature_encoder, 'reset_parameters'):
            self.feature_encoder.reset_parameters()
        self.type_emb.reset_parameters()
        self.hop_emb.reset_parameters()
        self.time_enc.reset_parameters()
        self.time_lin.reset_parameters()
        if hasattr(self.pe_encoder, 'reset_parameters'):
            self.pe_encoder.reset_parameters()
        self.lin.reset_parameters()

    def forward(
        self,
        feat_dict: Dict[str, Any],
        node_idx: Tensor,
        node_type: Tensor,
        hop: Tensor,
        rel_time: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        r"""Forward pass.

        Args:
            feat_dict (Dict[str, Any]): The per-node-type inputs to
                :obj:`feature_encoder`, *e.g.*, a dictionary holding a
                :class:`torch_frame.TensorFrame` for each node type.
            node_idx (torch.Tensor): The index of each token within its node
                type, *i.e.*, the row in the corresponding output of
                :obj:`feature_encoder`, of shape :obj:`[B, K]`.
            node_type (torch.Tensor): The node type ID of each token, of shape
                :obj:`[B, K]`.
            hop (torch.Tensor): The hop distance of each token from its seed
                node, of shape :obj:`[B, K]`.
                Filler nodes are denoted by :obj:`max_hops + 1`.
            rel_time (torch.Tensor): The time of each seed node minus the time
                of each of its tokens in seconds, of shape :obj:`[B, K]`.
            edge_index (torch.Tensor): The edge indices of the sampled
                subgraphs, where token :obj:`k` of seed node :obj:`b` is
                denoted by node index :obj:`b * K + k`.
        """
        if node_idx.dim() != 2:
            raise ValueError(f"Expected 'node_idx' to be two-dimensional "
                             f"(got {node_idx.dim()} dimensions)")
        for name, value in [('node_type', node_type), ('hop', hop),
                            ('rel_time', rel_time)]:
            if value.size() != node_idx.size():
                raise ValueError(f"Expected '{name}' to have shape "
                                 f"{tuple(node_idx.size())} "
                                 f"(got {tuple(value.size())})")

        B, K = node_idx.size()

        x_feat = self._gather_features(feat_dict, node_idx, node_type)
        x_type = self.type_emb(node_type)
        x_hop = self.hop_emb(hop)

        days = rel_time.to(self.time_lin.weight.dtype) / SECONDS_PER_DAY
        x_time = self.time_lin(self.time_enc(days.view(-1))).view(B, K, -1)

        batch = torch.arange(B, device=node_idx.device).repeat_interleave(K)
        x_pe = self.pe_encoder(edge_index, B * K, batch=batch, batch_size=B)
        x_pe = x_pe.view(B, K, -1)

        out = torch.cat([x_feat, x_type, x_hop, x_time, x_pe], dim=-1)
        return self.lin(out)

    def _gather_features(
        self,
        feat_dict: Dict[str, Any],
        node_idx: Tensor,
        node_type: Tensor,
    ) -> Tensor:
        x_dict = self.feature_encoder(feat_dict)

        out: Optional[Tensor] = None
        for i, key in enumerate(self.node_types):
            if key not in x_dict:
                if bool((node_type == i).any()):
                    raise ValueError(f"'feature_encoder' did not return "
                                     f"features for node type '{key}'")
                continue
            x = x_dict[key]
            if x.dim() != 2 or x.size(-1) != self.channels:
                raise ValueError(f"Expected 'feature_encoder' to return "
                                 f"features of shape [*, {self.channels}] for "
                                 f"node type '{key}' "
                                 f"(got {tuple(x.size())})")
            if out is None:
                out = x.new_zeros(*node_idx.size(), self.channels)
            mask = node_type == i
            out[mask] = x[node_idx[mask]]

        if out is None:
            raise ValueError(f"'feature_encoder' did not return features for "
                             f"any node type in {self.node_types}")
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.channels}, '
                f'num_node_types={len(self.node_types)}, '
                f'max_hops={self.max_hops})')
