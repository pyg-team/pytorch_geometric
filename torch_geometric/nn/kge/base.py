from typing import Tuple

import torch
from torch import Tensor
from torch.nn import Embedding
from tqdm import tqdm


class KGEModel(torch.nn.Module):
    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        hidden_channels: int,
        sparse: bool = False,
    ):
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

    def forward(self, head: Tensor, rel: Tensor, tail: Tensor) -> Tensor:
        """"""
        raise NotImplementedError

    def loss(self, head: Tensor, rel: Tensor, pos_tail: Tensor,
             neg_tail: Tensor) -> Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def test(self, head: Tensor, rel: Tensor, tail: Tensor, batch_size: int,
             filtered: bool = False, k: int = 10) -> Tuple[float, float]:

        if filtered:
            raise NotImplementedError("'Filtered' inference not yet supported")

        mean_ranks = []
        hits_at_k = []

        for i in tqdm(range(head.numel())):
            h, r, t = head[i], rel[i], tail[i]

            scores = []
            heads = torch.arange(self.num_nodes, device=head.device)
            for hs in heads.split(batch_size):
                scores.append(self(hs, r, t))
            rank = int((torch.cat(scores).argsort() == h).nonzero().view(-1))
            mean_ranks.append(rank)
            hits_at_k.append(rank < k)

            scores = []
            tails = torch.arange(self.num_nodes, device=tail.device)
            for ts in tails.split(batch_size):
                scores.append(self(h, r, ts))
            rank = int((torch.cat(scores).argsort() == t).nonzero().view(-1))
            mean_ranks.append(rank)
            hits_at_k.append(rank < k)

        mean_rank = float(torch.tensor(mean_ranks, dtype=torch.float).mean())
        hits_at_k = int(torch.tensor(hits_at_k).sum()) / len(hits_at_k)

        return mean_rank, hits_at_k

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.num_nodes}, '
                f'num_relations={self.num_relations}, '
                f'hidden_channels={self.hidden_channels})')
