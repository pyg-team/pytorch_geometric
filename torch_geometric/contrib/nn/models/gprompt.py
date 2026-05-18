from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class GraphAdapter(nn.Module):
    r"""Graph Adapter from the
    `Prompt-based Node Feature Extractor for Few-shot Learning on
    Text-Attributed Graphs (Huang et al., 2023)
    <https://arxiv.org/pdf/2309.02848>` _ paper.

    This module augments masked-token representations from a pre-trained
    language model with information from neighboring nodes in a
    text-attributed graph.

    It implements the gating mechanism and neighbor aggregation described
    in Section 3.2 of the paper.

    Args:
        hidden_channels (int): Dimensionality of masked-token hidden states
            :math:`\hat{h}_{i,k}`.
        node_channels (int): Dimensionality of node / sentence embeddings
            :math:`z_i`.
        attn_channels (int, optional): Dimensionality of attention
            projections :math:`W_q, W_k`. If set to :obj:`None`, defaults
            to :obj:`node_channels`.
        mlp_hidden_channels (int, optional): Hidden size of the MLP
            :math:`g(\cdot)` projecting node embeddings into the same space
            as :math:`\hat{h}_{i,k}`. If set to :obj:`None`, defaults to
            :obj:`node_channels`.
    """
    def __init__(
        self,
        hidden_channels: int,
        node_channels: int,
        attn_channels: int | None = None,
        mlp_hidden_channels: int | None = None,
    ) -> None:
        super().__init__()

        if attn_channels is None:
            attn_channels = node_channels
        if mlp_hidden_channels is None:
            mlp_hidden_channels = node_channels

        self.hidden_channels = hidden_channels
        self.node_channels = node_channels
        self.attn_channels = attn_channels
        self.mlp_hidden_channels = mlp_hidden_channels

        # Projections used in the neighbor gate a_ij:
        self.q_proj = nn.Linear(node_channels, attn_channels)
        self.k_proj = nn.Linear(node_channels, attn_channels)

        # MLP g(z_j) mapping node embeddings to the same space as h_mask:
        self.g_mlp = nn.Sequential(
            nn.Linear(node_channels, mlp_hidden_channels),
            nn.ReLU(),
            nn.Linear(mlp_hidden_channels, hidden_channels),
        )

    def forward(
        self,
        h_mask: Tensor,
        z: Tensor,
        edge_index: Tensor,
        node_ids: Tensor,
        predictor: nn.Module | Callable[[Tensor], Tensor],
    ) -> tuple[Tensor, Tensor | None]:
        r"""Runs the graph adapter.

        Args:
            h_mask (Tensor): Hidden states of masked tokens of shape
                :obj:`[M, hidden_channels]`.
            z (Tensor): Node / sentence embeddings of shape
                :obj:`[N, node_channels]`.
            edge_index (Tensor): Graph connectivity (src, dest) with
                shape :obj:`[2, E]`. edge_index[1] will correspond to the
                "center" nodes whose masked tokens we'll be predicting based
                on their neighbors in edge_index[0]
            node_ids (Tensor): Node index for each masked token of shape
                :obj:`[M]`, mapping masked tokens to graph nodes.
            predictor (Module or Callable): The language model's prediction
                head :math:`f_{\text{LM}}` that maps hidden states of shape
                :obj:`[..., hidden_channels]` to vocabulary logits
                :obj:`[..., vocab_size]`.

        Returns:
            (Tensor, Tensor): The first element is the per-edge logits
            and the second element are the masked-token indices of shape
            :obj:`[masked_edges]`.
        """
        device = h_mask.device
        src, dst = edge_index
        M, E = h_mask.size(0), src.numel()
        assert node_ids.dim() == 1 and len(node_ids) == M

        # Precompute attention projections q_i, k_j for all nodes.
        q = self.q_proj(z)  # [N, A]
        k = self.k_proj(z)  # [N, A]

        # Get the masked token id's for each node
        node_to_masked_dict: dict[int, list[int]] = {}
        for m, n in enumerate(node_ids.tolist()):
            if n not in node_to_masked_dict:
                node_to_masked_dict[n] = []
            node_to_masked_dict[n].append(m)

        # For each edge (src, dst) where dst has masked tokens,
        # we create one pair per token.
        masked_idx_list: list[tuple[int, int]] = []
        for e in range(E):
            center = int(dst[e].item())
            if center in node_to_masked_dict:
                for m in node_to_masked_dict[center]:
                    masked_idx_list.append((m, e))

        masked_idx_tensor = torch.tensor(
            [m for (m, _) in masked_idx_list],
            device=device,
            dtype=torch.long,
        )
        edge_idx_tensor = torch.tensor(
            [e for (_, e) in masked_idx_list],
            device=device,
            dtype=torch.long,
        )

        # For each (masked token m, edge e), get center node i and neighbor j:
        center_nodes = node_ids[masked_idx_tensor]  # [masked_edges]
        neighbor_nodes = src[edge_idx_tensor]  # [masked_edges]

        # Gate a_ij = sigmoid( (z_i W_q) (z_j W_k)^T ).
        q_i = q[center_nodes]  # [masked_edges, A]
        k_j = k[neighbor_nodes]  # [masked_edges, A]
        a_ij = torch.sigmoid(
            (q_i * k_j).sum(dim=-1, keepdim=True))  # [masked_edges, 1]

        # Neighbor influence:
        h_i = h_mask[masked_idx_tensor]  # [masked_edges, H]
        g_zj = self.g_mlp(z[neighbor_nodes])  # [masked_edges, H]
        h_tilde = a_ij * h_i + (1.0 - a_ij) * g_zj  # [masked_edges, H]

        # Apply pre-trained language model prediction head f_LM:
        edge_logits = predictor(h_tilde)  # [masked_edges, V]
        return edge_logits, masked_idx_tensor

    @staticmethod
    def pool_outputs(
        edge_logits: Tensor,
        masked_idx: Tensor,
        num_masked_tokens: int,
        use_log_probs: bool,
    ) -> Tensor:
        r"""Computes pooled probabilities per masked token by either taking the
        arithmetic mean of each masked token's neighbors' probabilities or
        taking the arithmetic mean of their log probs (corresponding to the
        geometric mean of the probabilities that is described in section 3.3
        in the paper). The paper formulates it in terms of the arithmetic mean
        but mentions that it is better to use the geometric mean when training
        (which is implicitly done in our loss function). We allow for both
        options here.

        Args:
            edge_logits (Tensor): Per-edge logits of shape
                :obj:`[masked_edges, vocab_size]`.
            masked_idx (Tensor): Index of the masked token for each edge of
                shape :obj:`[masked_edges]`.
            num_masked_tokens (int): Total number of masked tokens
                :math:`M`. The reason to have this as an explicit argument
                instead of inferring it from masked_idx is to handle cases
                where some masked tokens belong to nodes with no neighbors.
            use_log_probs (bool): If this is true, we will return the
            arithmetic mean of the log probabilities - i.e. if we take the exp
            of this, we will get the geometric mean of the probabilities.
            Otherwise, we will return the arithmetic mean of the probabilities
            directly.

        Returns:
            Tensor: output of shape
            :obj:`[M, vocab_size]` - if use_log_probs is True, this will
            represent the log-probabilities and otherwise this will represent
            the probabilities; if a masked token has no edges, we return
            negative inf in the case of log probs for that entry
            and 0 in the case of probabilities.
        """
        if use_log_probs:
            unpooled_outputs = F.log_softmax(edge_logits,
                                             dim=-1)  # [masked_edges, V]
        else:
            unpooled_outputs = F.softmax(edge_logits, dim=-1)
        pooled_sums = torch.zeros(
            num_masked_tokens,
            unpooled_outputs.shape[-1],
            device=unpooled_outputs.device,
            dtype=unpooled_outputs.dtype,
        )
        counts = torch.zeros(
            num_masked_tokens,
            device=unpooled_outputs.device,
            dtype=unpooled_outputs.dtype,
        )
        pooled_sums.index_add_(0, masked_idx,
                               unpooled_outputs)  # [num_masked_tokens, V]
        counts.index_add_(
            0,
            masked_idx,
            torch.ones(
                unpooled_outputs.shape[-2],
                device=unpooled_outputs.device,
                dtype=unpooled_outputs.dtype,
            ),
        )
        pooled_outputs = unpooled_outputs.new_full(
            (num_masked_tokens, unpooled_outputs.size(-1)),
            float("-inf") if use_log_probs else 0.0,
        )
        mask = counts > 0
        pooled_outputs[mask] = pooled_sums[mask] / counts[mask].unsqueeze(-1)
        return pooled_outputs

    @staticmethod
    def loss(
        edge_logits: Tensor,
        masked_idx: Tensor,
        target_ids: Tensor,
    ) -> Tensor:
        r"""Computes the graph-adapter loss as described in Eq. (8) of the
        paper.

        The loss corresponds to averaging the cross-entropy over neighbors
        for each masked token, which is equivalent to the negative log of
        the geometric mean of the probabilities, as formulated in the paper.

        Args:
            edge_logits (Tensor): Per-edge logits of shape
                :obj:`[masked_edges, vocab_size]`, where :obj:`masked_edges`
                is the number of edges summed over all masked tokens (i.e. the
                length of all (masked-token, neighbor) pairs).
            masked_idx (Tensor): Index of the masked token for each edge of
                shape :obj:`[masked_edges]`.
            target_ids (Tensor): Ground-truth token ids for each masked token
                of shape :obj:`[M]`.

        Returns:
            Tensor: Scalar loss.
        """
        # We compute the negative log loss with respect to all of the
        # neighbors of each masked central node. Then we scale it down by the
        # number of neighbors for that node.
        edge_targets = target_ids[masked_idx]  # [masked_edges]
        edge_losses = F.cross_entropy(edge_logits, edge_targets,
                                      reduction="none")  # [masked_edges]
        numerator = torch.zeros(target_ids.size(0), device=edge_logits.device)
        denominator = torch.zeros(target_ids.size(0),
                                  device=edge_logits.device)
        numerator.scatter_add_(0, masked_idx, edge_losses)
        denominator.scatter_add_(
            0, masked_idx, torch.ones_like(masked_idx,
                                           dtype=edge_losses.dtype))
        if not denominator.any():
            return torch.tensor(0.0)
        return (numerator[denominator > 0] /
                denominator[denominator > 0]).mean()
