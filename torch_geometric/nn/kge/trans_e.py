import math

import torch
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn.kge import KGEModel


class TransE(KGEModel):
    def __init__(self, num_nodes: int, num_relations: int,
                 hidden_channels: int, p_norm: float = 1.0,
                 sparse: bool = False):
        self.p_norm = p_norm
        super().__init__(num_nodes, num_relations, hidden_channels, sparse)

    @torch.no_grad()
    def reset_parameters(self):
        bound = 6. / math.sqrt(self.hidden_channels)
        torch.nn.init.uniform_(self.node_emb.weight, -bound, bound)
        torch.nn.init.uniform_(self.rel_emb.weight, -bound, bound)
        F.normalize(self.rel_emb.weight, p=self.p_norm, dim=-1,
                    out=self.rel_emb.weight)

    def forward(self, edge_index: Tensor, edge_type: Tensor) -> Tensor:
        src, dst = edge_index[0], edge_index[1]

        src = self.node_emb(src)
        dst = self.node_emb(dst)
        rel = self.rel_emb(edge_type)

        src = F.normalize(src, p=self.p_norm, dim=-1)
        dst = F.normalize(dst, p=self.p_norm, dim=-1)

        return src.add_(rel).sub_(dst).norm(p=self.p_norm, dim=-1).view(-1)

    def loss(self, edge_index: Tensor, edge_type: Tensor, pos_mask: Tensor,
             margin: float = 1.0) -> Tensor:
        score = self(edge_index, edge_type)
        pos_score, neg_score = score[pos_mask], score[~pos_mask]
        return torch.clamp(pos_score.add_(margin).sub_(neg_score), 0.).mean()
