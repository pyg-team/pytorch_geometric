import math

import torch
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn.kge import KGEModel


class TransE(KGEModel):
    r"""TODO"""
    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        hidden_channels: int,
        margin: float = 1.0,
        p_norm: float = 1.0,
        sparse: bool = False,
    ):
        self.p_norm = p_norm
        self.margin = margin
        super().__init__(num_nodes, num_relations, hidden_channels, sparse)

    def reset_parameters(self):
        bound = 6. / math.sqrt(self.hidden_channels)
        torch.nn.init.uniform_(self.node_emb.weight, -bound, bound)
        torch.nn.init.uniform_(self.rel_emb.weight, -bound, bound)
        F.normalize(self.rel_emb.weight.data, p=self.p_norm, dim=-1,
                    out=self.rel_emb.weight.data)

    def forward(self, head: Tensor, rel: Tensor, tail: Tensor) -> Tensor:
        head_emb = self.node_emb(head)
        rel_emb = self.rel_emb(rel)
        tail_emb = self.node_emb(tail)

        head_emb = F.normalize(head_emb, p=self.p_norm, dim=-1)
        tail_emb = F.normalize(tail_emb, p=self.p_norm, dim=-1)

        return ((head_emb + rel_emb) - tail_emb).norm(p=self.p_norm, dim=-1)

    def loss(self, head: Tensor, rel: Tensor, tail: Tensor) -> Tensor:
        pos_score = self(head, rel, tail)

        # Random shuffle either `head` or `tail`:
        rnd = torch.randint(self.num_nodes, head.size(), device=head.device)
        head = head.clone()
        head[:head.numel() // 2] = rnd[:head.numel() // 2]
        tail = tail.clone()
        tail[tail.numel() // 2:] = rnd[tail.numel() // 2:]

        neg_score = self(head, rel, tail)

        return (self.margin + pos_score - neg_score).relu().mean()
