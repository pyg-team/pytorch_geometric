from typing import Tuple

import torch
from torch import Tensor
from torch.nn import Embedding
from tqdm import tqdm

from torch_geometric.nn.kge.loader import KGTripletLoader


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

    def loader(self, head_index: Tensor, rel_type: Tensor, tail_index: Tensor,
               **kwargs) -> Tensor:
        r"""Returns a mini-batch loader to sample triplets from
        :obj:`(head_index, rel_type, tail_index)`."""
        return KGTripletLoader(head_index, rel_type, tail_index, **kwargs)

    def forward(self, head_index: Tensor, rel_type: Tensor,
                tail_index: Tensor) -> Tensor:
        """"""
        raise NotImplementedError

    def loss(self, head_index: Tensor, rel_type: Tensor,
             tail_index: Tensor) -> Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def random_sample(self, head_index: Tensor, rel_type: Tensor,
                      tail_index: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # Random sample either `head_index` or `tail_index` (but not both):
        num_negatives = head_index.numel() // 2
        rnd_index = torch.randint(self.num_nodes, head_index.size(),
                                  device=head_index.device)

        head_index = head_index.clone()
        head_index[:num_negatives] = rnd_index[:num_negatives]
        tail_index = tail_index.clone()
        tail_index[num_negatives:] = rnd_index[num_negatives:]

        return head_index, rel_type, tail_index

    @torch.no_grad()
    def test(self, head_index: Tensor, rel_type: Tensor, tail_index: Tensor,
             batch_size: int, k: int = 10) -> Tuple[float, float]:
        r"""Evaluates the model quality by computing mean rank and hits@`k`
        across all possible tail entities.

        Args:
            head_index (torch.Tensor): The head indices.
            rel_type (torch.Tensor): The relation type.
            tail_index (torch.Tensor): The tail indices.
            batch_size (int): The batch size to use for evaluating.
            k (int, optional): The `k` in hits@`k`. (default: :obj:`10`)
        """
        mean_ranks, hits_at_k = [], []
        for i in tqdm(range(head_index.numel())):
            h, r, t = head_index[i], rel_type[i], tail_index[i]

            scores = []
            tail_indices = torch.arange(self.num_nodes, device=t.device)
            for ts in tail_indices.split(batch_size):
                scores.append(self(h.expand_as(ts), r.expand_as(ts), ts))
            rank = int((torch.cat(scores).argsort(
                descending=True) == t).nonzero().view(-1))
            mean_ranks.append(rank)
            hits_at_k.append(rank < k)

        mean_rank = float(torch.tensor(mean_ranks, dtype=torch.float).mean())
        hits_at_k = int(torch.tensor(hits_at_k).sum()) / len(hits_at_k)

        return mean_rank, hits_at_k

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.num_nodes}, '
                f'num_relations={self.num_relations}, '
                f'hidden_channels={self.hidden_channels})')
