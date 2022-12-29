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

    def get_node_embedding(self, index: Tensor):
        node_emb = self.node_emb(index)
        node_emb = F.normalize(node_emb, p=self.p_norm, dim=-1)
        return node_emb

    def score(self, head_emb: Tensor, rel_emb: Tensor,
              tail_emb: Tensor) -> Tensor:
        return ((head_emb + rel_emb) - tail_emb).norm(p=self.p_norm, dim=-1)

    def forward(self, head: Tensor, rel: Tensor, tail: Tensor) -> Tensor:
        head_emb = self.get_node_embedding(head)
        rel_emb = self.rel_emb(rel)
        tail_emb = self.get_node_embedding(tail)

        return self.score(head_emb, rel_emb, tail_emb)

    def loss(self, head: Tensor, rel: Tensor, tail: Tensor) -> Tensor:
        print(head)
        print('rel', rel)
        print(tail)
        print(head.max())
        print(rel.max())
        print(tail.max())
        head_emb = self.get_node_embedding(head)
        rel_emb = self.rel_emb(rel)
        tail_emb = self.get_node_embedding(tail)

        neg_tail = torch.randint(self.num_nodes, tail.size(),
                                 device=tail.device)
        print(neg_tail)
        neg_tail_emb = self.get_node_embedding(neg_tail)

        pos_score = self.score(head_emb, rel_emb, tail_emb)
        neg_score = self.score(head_emb, rel_emb, neg_tail_emb)

        return (self.margin + pos_score - neg_score).relu().mean()
