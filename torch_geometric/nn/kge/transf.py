import math

import torch
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn.kge import KGEModel

class TransF(KGEModel):
    r"""The TransF model from the "Knowledge Graph Embedding by
    Flexible Translation" cccccbchdufkjvlkejkfhftultfgcigvrdugvpaper.

    :class:`TransF` introduces a flexible translation mechanism by dynamically
    scaling the relation vector based on head and tail entity embeddings,
    resulting in:

    .. math::
        \mathbf{e}_h + f(\mathbf{e}_h, \mathbf{e}_t, \mathbf{e}_r)
    .. math::
        \cdot \mathbf{e}_r \approx \mathbf{e}_t

    where :math:`f` is a dynamic scaling function:

    .. math::
        f(\mathbf{e}_h, \mathbf{e}_t, \mathbf{e}_r) =
    .. math::
        \sigma((\mathbf{e}_h \odot \mathbf{e}_t) \cdot \mathbf{e}_r)

    This results in the scoring function:

    .. math::
        d(h, r, t) = - \| \mathbf{e}_h +
    .. math::
        f(\mathbf{e}_h, \mathbf{e}_t, \mathbf{e}_r) \cdot
    .. math::
        \mathbf{e}_r - \mathbf{e}_t \|_p

    .. note::
        For an example of using the :class:`TransF` model, see
        `examples/kge_fb15k_237.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        kge_fb15k_237.py>`_.

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
    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        hidden_channels: int,
        margin: float = 1.0,
        p_norm: float = 1.0,
        sparse: bool = False,
    ):
        super().__init__(num_nodes, num_relations, hidden_channels, sparse)

        self.p_norm = p_norm
        self.margin = margin

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
        rel = self.rel_emb(rel_type)
        tail = self.node_emb(tail_index)

        head = F.normalize(head, p=self.p_norm, dim=-1)
        tail = F.normalize(tail, p=self.p_norm, dim=-1)

        # Flexible scaling function based on head, relation, and tail
        scaling_factor = torch.sigmoid((head * tail).sum(dim=-1, keepdim=True))
        adjusted_rel = scaling_factor * rel

        # Calculate *negative* TransF norm:
        return -((head + adjusted_rel) - tail).norm(p=self.p_norm, dim=-1)

    def loss(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
    ) -> Tensor:

        pos_score = self(head_index, rel_type, tail_index)
        neg_score = self(*self.random_sample(head_index, rel_type, tail_index))

        return F.margin_ranking_loss(
            pos_score,
            neg_score,
            target=torch.ones_like(pos_score),
            margin=self.margin,
        )
