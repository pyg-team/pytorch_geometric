import math
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Embedding

from torch_geometric.nn.kge import KGEModel
from torch_geometric.nn.kge.loader import KGTripletLoader


class TransH(KGEModel):
    r"""The TransH model from the `"Knowledge Graph Embedding by Translating
    on Hyperplanes" <https://www.researchgate.net/publication/
    319207032_Knowledge_Graph_Embedding_by_Translating_on_Hyperplanes>`_ paper.

    :class:`TransH` models relations as a translation from projected head to
    tail entities such that

    .. math::
        \mathbf{e}_{h_{\perp}} + \mathbf{d}_r \approx \mathbf{e}_{t_{\perp}},

    resulting in the scoring function:

    .. math::
        - || (\mathbf{e_h} - \mathbf{w}_r^T\mathbf{e_hw}_r) + \mathbf{d}_r -
        (\mathbf{e_t} - \mathbf{w}_r^T\mathbf{e_tw}_r) ||_2^2
    .. note::

        For an example of using the :class:`TransH` model, see
        # `examples/kge_fb15k_237.py
        # <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        # kge_fb15k_237.py>`_.

    Args:
        num_nodes (int): The number of nodes/entities in the graph.
        num_relations (int): The number of relations in the graph.
        hidden_channels (int): The hidden embedding size.
        margin (int, optional): The margin of the ranking loss.
            (default: :obj:`1.0`)
        p_norm (int, optional): The order embedding and distance normalization.
            (default: :obj:`1.0`)
        sparse (bool, optional): If set to :obj:`True`, gradients w.r.t. to the
            embedding matrices will be sparse. (default: :obj:`False`)
        bernoulli (bool, optional): Random sampling technique.
            If set to :obj:`True`, sample negative labels using a Bernoulli
            distribution. (default: :obj:`False`)
    """
    def __init__(self, num_nodes: int, num_relations: int,
                 hidden_channels: int, margin: float = 1.0,
                 p_norm: float = 2.0, sparse: bool = False,
                 bernoulli: bool = False):
        super().__init__(num_nodes, num_relations, hidden_channels, sparse)
        self.bernoulli = bernoulli
        self.head_corruption_probs = torch.zeros(num_relations)

        self.p_norm = p_norm
        self.margin = margin

        self.w_rel_emb = Embedding(num_relations, hidden_channels,
                                   sparse=sparse)
        self.d_rel_emb = Embedding(num_relations, hidden_channels,
                                   sparse=sparse)

        self.reset_parameters()

    def reset_parameters(self):
        bound = 6. / math.sqrt(self.hidden_channels)
        torch.nn.init.uniform_(self.node_emb.weight, -bound, bound)
        torch.nn.init.uniform_(self.w_rel_emb.weight, -bound, bound)
        torch.nn.init.uniform_(self.d_rel_emb.weight, -bound, bound)
        F.normalize(self.w_rel_emb.weight.data, p=self.p_norm, dim=-1,
                    out=self.w_rel_emb.weight.data)
        F.normalize(self.d_rel_emb.weight.data, p=self.p_norm, dim=-1,
                    out=self.d_rel_emb.weight.data)

    def forward(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
    ) -> Tensor:

        head = self.node_emb(head_index)
        tail = self.node_emb(tail_index)

        w_rel = F.normalize(self.w_rel_emb(rel_type), p=self.p_norm)
        d_rel = self.d_rel_emb(rel_type)

        proj_head = head - (w_rel * head).sum(dim=1).unsqueeze(dim=1) * w_rel
        proj_tail = tail - (w_rel * tail).sum(dim=1).unsqueeze(dim=1) * w_rel

        proj_head = F.normalize(proj_head, p=self.p_norm, dim=-1)
        proj_tail = F.normalize(proj_tail, p=self.p_norm, dim=-1)

        # Calculate *negative* TransH norm:
        return -((
            (proj_head + d_rel) - proj_tail).norm(p=self.p_norm, dim=-1))**2

    def loss(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
    ) -> Tensor:

        pos_score = self(head_index, rel_type, tail_index)
        if self.bernoulli:
            neg_score = self(*self.random_sample_bernoulli(
                head_index, rel_type, tail_index))
        else:
            neg_score = self(
                *self.random_sample(head_index, rel_type, tail_index))

        return F.margin_ranking_loss(
            pos_score,
            neg_score,
            target=torch.ones_like(pos_score),
            margin=self.margin,
        )

    def loader(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
        **kwargs,
    ) -> Tensor:

        if self.bernoulli:
            self.compute_corrupt_probs(head_index, rel_type, tail_index)
        return KGTripletLoader(head_index, rel_type, tail_index, **kwargs)

    def random_sample_bernoulli(
            self, head_index: Tensor, rel_type: Tensor,
            tail_index: Tensor) -> Tuple[Tensor, Tensor, Tensor]:

        head_index = head_index.clone()
        tail_index = tail_index.clone()

        probs = self.head_corruption_probs[rel_type]
        corrupt_heads = torch.bernoulli(probs).bool()
        corrupt_tails = ~corrupt_heads

        head_index[corrupt_heads] = torch.randint(0, self.num_nodes,
                                                  (corrupt_heads.sum(), ))
        tail_index[corrupt_tails] = torch.randint(0, self.num_nodes,
                                                  (corrupt_tails.sum(), ))

        return head_index, rel_type, tail_index

    def compute_corrupt_probs(
            self, head_index: Tensor, rel_type: Tensor,
            tail_index: Tensor) -> Tuple[Tensor, Tensor, Tensor]:

        for r in range(self.num_relations):
            mask = (rel_type == r).bool()
            _, tail_count = torch.unique(head_index[mask],
                                         return_counts=True, dim=0)
            _, head_count = torch.unique(tail_index[mask],
                                         return_counts=True, dim=0)
            tph = torch.mean(tail_count, dtype=torch.float64)
            hpt = torch.mean(head_count, dtype=torch.float64)
            self.head_corruption_probs[r] = tph / (tph + hpt)
