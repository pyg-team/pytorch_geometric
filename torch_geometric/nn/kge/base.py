from typing import Tuple

import torch
from torch import Tensor
from torch.nn import Embedding
from tqdm import tqdm

from .loader import DataLoader


class KGEModel(torch.nn.Module):
    def __init__(self, num_nodes: int, num_relations: int,
                 hidden_channels: int, sparse: bool = False):
        super().__init__()

        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.hidden_channels = hidden_channels

        self.node_emb = Embedding(num_nodes, hidden_channels, sparse=sparse)
        self.rel_emb = Embedding(num_relations, hidden_channels, sparse=sparse)

        self.reset_parameters()

    def reset_parameters(self):
        self.node_emb.reset_parameters()
        self.rel_emb.reset_parameters()

    def forward(self, edge_index: Tensor, edge_type: Tensor) -> Tensor:
        """"""
        raise NotImplementedError

    def loss(self, edge_index: Tensor, edge_type: Tensor,
             pos_mask: Tensor) -> Tensor:
        raise NotImplementedError

    def loader(self, edge_index: Tensor, edge_type: Tensor,
               add_negative_samples: bool = True, **kwargs):
        return DataLoader(self.num_nodes, edge_index, edge_type,
                          add_negative_samples, **kwargs)

    @torch.no_grad()
    def test(self, edge_index: Tensor, edge_type: Tensor, batch_size: int,
             filtered: bool = False, k: int = 10) -> Tuple[float, float]:

        if filtered:
            raise NotImplementedError("'Filtered' inference not yet supported")

        device = self.node_emb.weight.device

        mean_ranks = []
        hits_at_k = []

        for j in tqdm(range(edge_index.size(1))):
            (src, dst), rel = edge_index[:, j], edge_type[j]

            src_edge_index = torch.stack([
                torch.arange(self.num_nodes, device=src.device),
                dst.expand(self.num_nodes),
            ], dim=0)
            src_edge_type = rel.expand(self.num_nodes).to(device)

            src_scores = []
            for i in range(0, self.num_nodes, batch_size):
                score = self(src_edge_index[:, i:i + batch_size].to(device),
                             src_edge_type[i:i + batch_size].to(device))
                src_scores.append(score)
            src_score = torch.cat(src_scores, dim=0)
            rank = (src_score.argsort() == src).nonzero(as_tuple=False)
            rank = int(rank.view(-1)[0])
            mean_ranks.append(rank)
            hits_at_k.append(rank < k)

            dst_edge_index = torch.stack([
                src.expand(self.num_nodes),
                torch.arange(self.num_nodes, device=dst.device),
            ], dim=0)
            dst_edge_type = rel.expand(self.num_nodes).to(device)

            dst_scores = []
            for i in range(0, self.num_nodes, batch_size):
                score = self(dst_edge_index[:, i:i + batch_size].to(device),
                             dst_edge_type[i:i + batch_size].to(device))
                dst_scores.append(score)
            dst_score = torch.cat(dst_scores, dim=0)
            rank = (dst_score.argsort() == dst).nonzero(as_tuple=False)
            rank = int(rank.view(-1)[0])
            mean_ranks.append(rank)
            hits_at_k.append(rank < k)

        mean_rank = float(torch.tensor(mean_ranks, dtype=torch.float).mean())
        hits_at_k = int(torch.tensor(hits_at_k).sum()) / len(hits_at_k)

        return mean_rank, hits_at_k

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.num_nodes}, '
                f'num_relations={self.num_relations}, '
                f'hidden_channels={self.hidden_channels})')
