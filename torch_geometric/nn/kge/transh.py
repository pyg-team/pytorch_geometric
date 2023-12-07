import math
import random
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Embedding

from torch_geometric.nn.kge import KGEModel


class TransH(KGEModel):
    r"""The TransH model from the `"Knowledge Graph Embedding by Translating
    on Hyperplanes" <https://www.researchgate.net/publication/
    319207032_Knowledge_Graph_Embedding_by_Translating_on_Hyperplanes>`_ paper.

    :class:`TransH` models relations as a translation from projected head to tail
    entities such that

    .. math::
        \mathbf{e}_{h_{\perp}} + \mathbf{d}_r \approx \mathbf{e}_{t_{\perp}},

    resulting in the scoring function:

    .. math::
        - || (\mathbf{e_h} - \mathbf{w}_r^T\mathbf{e_hw}_r) + \mathbf{d}_r - (\mathbf{e_t} - \mathbf{w}_r^T\mathbf{e_tw}_r) ||_2^2
    .. note::

        For an example of using the :class:`TransH` model, see #TODO.
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
    """
    def __init__(self, num_nodes: int, num_relations: int,
                 hidden_channels: int, margin: float = 1.0,
                 p_norm: float = 2.0, sparse: bool = False,
                 bernoulli: bool = True):
        super().__init__(num_nodes, num_relations, hidden_channels, sparse)
        self.bernoulli = bernoulli

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
        torch.nn.init.uniform_(self.rel_emb.weight, -bound, bound)
        F.normalize(self.rel_emb.weight.data, p=self.p_norm, dim=-1,
                    out=self.rel_emb.weight.data)

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

        # TODO: incorporate true/false using bernoulli or regular

        # TODO: check page 4 for adding soft constraints to loss
        pos_score = self(head_index, rel_type, tail_index)
        if self.bernoulli:
            neg_score = self(
                *self.sample_golden_triplets(head_index, rel_type, tail_index))
        else:
            neg_score = self(
                *self.random_sample(head_index, rel_type, tail_index))

        return F.margin_ranking_loss(
            pos_score,
            neg_score,
            target=torch.ones_like(pos_score),
            margin=self.margin,
        )

    # TODO: change function name to random_sample after verifying it works
    def sample_golden_triplets(
            self, head_index: Tensor, rel_type: Tensor,
            tail_index: Tensor) -> Tuple[Tensor, Tensor, Tensor]:

        #get distinct relations
        # unique_relations = torch.unique(rel_type)
        # head_corruption_prob = torch.zeros(rel_type.shape[0])
        # for i in range(unique_relations.shape[0]):
        #     r = unique_relations[i]
        #     mask = (rel_type == r).int()
        #     _, tail_count = torch.unique((head_index[mask]), return_counts=True, dim=0)
        #     _, head_count = torch.unique(tail_index[mask], return_counts=True, dim=0)
        #     tph = torch.mean(tail_count, dtype=torch.float64)
        #     hpt = torch.mean(head_count, dtype=torch.float64)
        #     prob_mask = torch.Tensor([tph / (tph + hpt) if val == 1 else 0 for val in mask])
        #     head_corruption_prob += prob_mask

        head_index = head_index.clone()
        tail_index = tail_index.clone()

        # head_index = torch.Tensor([random.randint(0, self.num_nodes - 1) if torch.bernoulli(head_corruption_prob) == 1 else head_index])
        # tail_index = torch.Tensor([random.randint(0, self.num_nodes - 1) if torch.bernoulli(head_corruption_prob) == 0 else tail_index])
        # for i in range(head_index.shape[0]):
        #     prob = self.head_corruption_probs[rel_type[i]]
        #     corrupt_head = torch.bernoulli(prob)
        #     if corrupt_head:
        #         # replace head with random head
        #         head_index[i] = random.randint(0, self.num_nodes - 1)
        #     else:
        #         # replace tail with random tail
        #         tail_index[i] = random.randint(0, self.num_nodes - 1)

        probs = torch.Tensor(
            [self.head_corruption_probs[r.item()] for r in rel_type])
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

        self.head_corruption_probs = {}
        for r in rel_type:
            r = r.item()
            if r not in self.head_corruption_probs:
                mask = (rel_type == r).int()
                _, tail_count = torch.unique((head_index[mask]),
                                             return_counts=True, dim=0)
                _, head_count = torch.unique(tail_index[mask],
                                             return_counts=True, dim=0)
                tph = torch.mean(tail_count, dtype=torch.float64)
                hpt = torch.mean(head_count, dtype=torch.float64)
                # prob_mask = torch.Tensor([tph / (tph + hpt) if val == 1 else 0 for val in mask])
                self.head_corruption_probs[r] = tph / (tph + hpt)

    #    torch.unique((head_index, rel_type, tail_index), return_counts=True)

    #    print('head:', head_index.shape)
    #    print('rel:', rel_type.shape)
    #    print(head_index)
    #    print('tail:', tail_index.shape)

    # hpt = head_index.numel() // tail_index.numel()  # average number of head entities per tail entity
    # tph = tail_index.numel() // head_index.numel() # average number of tail entities per head entity

    # corrupt_head_prob = torch.empty(self.num_relations)

    # for r in range(self.num_relations):

    # print('head_index size:', head_index.size)

    #bernoulli = torch.bernoulli(input=tph / (tph + hpt), out=)

    # if bernoulli:
    #     # replace head
    #     pass
    # else:
    #     # replace tail
    #     pass

    # FROM ORIGINAL RANDOM SAMPLE
    # # Random sample either `head_index` or `tail_index` (but not both):
    # num_negatives = head_index.numel() // 2
    # rnd_index = torch.randint(self.num_nodes, head_index.size(),
    #                           device=head_index.device)

    # head_index = head_index.clone()
    # head_index[:num_negatives] = rnd_index[:num_negatives]
    # tail_index = tail_index.clone()
    # tail_index[num_negatives:] = rnd_index[num_negatives:]
