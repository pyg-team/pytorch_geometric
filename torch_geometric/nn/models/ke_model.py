from typing import List, Tuple

import math
from tqdm import tqdm

import torch
from torch import Tensor
from torch.nn import Embedding
import torch.nn.functional as F


class KEModel(torch.nn.Module):
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


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, num_nodes: int, edge_index: Tensor, edge_type: Tensor,
                 add_negative_samples: bool = True, **kwargs):
        self.num_nodes = num_nodes
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.add_negative_samples = add_negative_samples
        super().__init__(range(edge_index.size(1)), collate_fn=self.sample,
                         **kwargs)

    def sample(self, batch: List[int]) -> Tuple[Tensor, Tensor, Tensor]:
        numel = len(batch)
        device = self.edge_index.device
        batch = torch.tensor(batch, device=device)

        edge_index = torch.empty((2, 2 * numel), dtype=torch.long,
                                 device=device)
        edge_type = torch.empty(2 * numel, dtype=torch.long, device=device)
        pos_mask = torch.empty(2 * numel, dtype=torch.bool, device=device)

        edge_index[0, :numel] = self.edge_index[0, batch]
        edge_index[1, :numel] = self.edge_index[1, batch]
        edge_index[0, numel:] = edge_index[0, :numel]
        edge_index[1, numel:] = edge_index[1, :numel]

        rnd = torch.randint(self.num_nodes, (numel, ), dtype=torch.long,
                            device=device)
        src_mask = torch.rand(numel, device=device) < 0.5

        edge_index[0, numel:][src_mask] = rnd[src_mask]
        edge_index[1, numel:][~src_mask] = rnd[~src_mask]

        edge_type[:numel] = self.edge_type[batch]
        edge_type[numel:] = edge_type[:numel]

        pos_mask[:numel] = True
        pos_mask[numel:] = False

        return edge_index, edge_type, pos_mask


class TransE(KEModel):
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
