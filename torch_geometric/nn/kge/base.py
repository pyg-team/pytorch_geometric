from typing import Tuple, List

import torch
from torch import Tensor
from torch.nn import Embedding
from tqdm import tqdm

from torch_geometric.nn.kge.loader import KGTripletLoader


class KGEModel(torch.nn.Module):
    r"""An abstract base class for implementing custom KGE models.

    Args:
        num_nodes (int): The number of nodes/entities in the graph.
        num_relations (int): The number of relations in the graph.
        hidden_channels (int): The hidden embedding size.
        sparse (bool, optional): If set to :obj:`True`, gradients w.r.t. to the
            embedding matrices will be sparse. (default: :obj:`False`)
    """

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

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.node_emb.reset_parameters()
        self.rel_emb.reset_parameters()

    def forward(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
    ) -> Tensor:
        r"""Returns the score for the given triplet.

        Args:
            head_index (torch.Tensor): The head indices.
            rel_type (torch.Tensor): The relation type.
            tail_index (torch.Tensor): The tail indices.
        """
        raise NotImplementedError

    def loss(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
    ) -> Tensor:
        r"""Returns the loss value for the given triplet.

        Args:
            head_index (torch.Tensor): The head indices.
            rel_type (torch.Tensor): The relation type.
            tail_index (torch.Tensor): The tail indices.
        """
        raise NotImplementedError

    def loader(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
        **kwargs,
    ) -> Tensor:
        r"""Returns a mini-batch loader that samples a subset of triplets.

        Args:
            head_index (torch.Tensor): The head indices.
            rel_type (torch.Tensor): The relation type.
            tail_index (torch.Tensor): The tail indices.
            **kwargs (optional): Additional arguments of
                :class:`torch.utils.data.DataLoader`, such as
                :obj:`batch_size`, :obj:`shuffle`, :obj:`drop_last`
                or :obj:`num_workers`.
        """
        return KGTripletLoader(head_index, rel_type, tail_index, **kwargs)

    @torch.no_grad()
    def test(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
        batch_size: int,
        k: int = 10,
        log: bool = True,
        filtered: bool = False,
        neighbors: List[List[List[int]]] = None
    ) -> Tuple[float, float, float]:
        r"""Evaluates the model quality by computing Mean Rank, MRR and
        Hits@:math:`k` across all possible tail entities.

        Args:
            head_index (torch.Tensor): The head indices.
            rel_type (torch.Tensor): The relation type.
            tail_index (torch.Tensor): The tail indices.
            batch_size (int): The batch size to use for evaluating.
            k (int, optional): The :math:`k` in Hits @ :math:`k`.
                (default: :obj:`10`)
            log (bool, optional): If set to :obj:`False`, will not print a
                progress bar to the console. (default: :obj:`True`)
            filtered (bool, optional): If set to :obj:`True`, does evaluation
                in the filtered setting, as described in
                [Bordes et al, 2013](https://dl.acm.org/doi/10.5555/2999792.2999923).
                This mode filters out all tails present in the training, validation,
                or test set for any (head,relation) from the candidate set of tails
                while computing the rank. (default: :obj:`False`)
            neighbors (List[List[List[int]]], optional): List of tails present in the
                training, validation or test set for any (head,relation) pair in the
                dataset. Specifically, if :math:`(h,r,t)` is present in the dataset,
                :math:`t` is present in :math:`neighbors[h][r]`. (default: :obj:`None`)
        """
        arange = range(head_index.numel())
        arange = tqdm(arange) if log else arange

        mean_ranks, reciprocal_ranks, hits_at_k = [], [], []
        for i in arange:
            h, r, t = head_index[i], rel_type[i], tail_index[i]

            scores = []
            tail_indices = torch.arange(self.num_nodes, device=t.device)
            for ts in tail_indices.split(batch_size):
                scores.append(self(h.expand_as(ts), r.expand_as(ts), ts))
            flattened_scores = torch.cat(scores)
            if filtered:
                curr_neighbours = neighbors[h.item()][r.item()]
                mask_indices = []
                for e_id in curr_neighbours:
                    if e_id == t.item():
                        # Do not filter out the gold answer
                        continue
                    mask_indices.append(e_id)
                mask_indices = torch.LongTensor(
                    mask_indices).to(head_index.device)
                flattened_scores.index_fill_(0, mask_indices, -100)
            rank = int((flattened_scores.argsort(
                descending=True) == t).nonzero().view(-1))
            mean_ranks.append(rank)
            reciprocal_ranks.append(1 / (rank + 1))
            hits_at_k.append(rank < k)

        mean_rank = float(torch.tensor(mean_ranks, dtype=torch.float).mean())
        mrr = float(torch.tensor(reciprocal_ranks, dtype=torch.float).mean())
        hits_at_k = int(torch.tensor(hits_at_k).sum()) / len(hits_at_k)

        return mean_rank, mrr, hits_at_k

    @torch.no_grad()
    def random_sample(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        r"""Randomly samples negative triplets by either replacing the head or
        the tail (but not both).

        Args:
            head_index (torch.Tensor): The head indices.
            rel_type (torch.Tensor): The relation type.
            tail_index (torch.Tensor): The tail indices.
        """
        # Random sample either `head_index` or `tail_index` (but not both):
        num_negatives = head_index.numel() // 2
        rnd_index = torch.randint(self.num_nodes, head_index.size(),
                                  device=head_index.device)

        head_index = head_index.clone()
        head_index[:num_negatives] = rnd_index[:num_negatives]
        tail_index = tail_index.clone()
        tail_index[num_negatives:] = rnd_index[num_negatives:]

        return head_index, rel_type, tail_index

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.num_nodes}, '
                f'num_relations={self.num_relations}, '
                f'hidden_channels={self.hidden_channels})')
