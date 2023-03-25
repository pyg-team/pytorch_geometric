import math

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Embedding

from torch_geometric.nn.kge import KGEModel


class RotatE(KGEModel):
    r"""The RotatE model from the `"RotatE: Knowledge Graph Embedding by
    Relational Rotation in Complex Space" <https://arxiv.org/abs/
    1902.10197>`_ paper.

    :class:`RotatE` models relations as a rotation in complex space
    from head to tail such that:

    .. math::
        \mathbf{e}_t = \mathbf{e}_h \circ \mathbf{e}_r

    Resulting in the scoring function:

    .. math::
        d(h, r, t) = - {\| \mathbf{e}_h \circ \mathbf{e}_r - \mathbf{e}_t \|}_p

    .. note::

        For an example of using the :class:`RotatE` model, see
        `examples/kge_fb15k_237.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        kge_fb15k_237.py>`_.

    Args:
        num_nodes (int): The number of nodes/entities in the graph.
        num_relations (int): The number of relations in the graph.
        hidden_channels (int): The hidden embedding size.
        margin (float, optional): The margin of the ranking loss.
        sparse (bool, optional): If set to :obj:`True`, gradients w.r.t. to
            the embedding matrices will be sparse. (default: :obj:`False`)
    """
    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        hidden_channels: int,
        margin: float = 1.0,
        sparse: bool = False,
    ):
        super().__init__(num_nodes, num_relations, hidden_channels, sparse)
        self.register_buffer('margin', torch.Tensor([margin]))
        self.node_emb_im = Embedding(num_nodes, hidden_channels, sparse=sparse)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.node_emb.weight)
        torch.nn.init.xavier_uniform_(self.node_emb_im.weight)
        torch.nn.init.uniform_(self.rel_emb.weight, 0, 2 * math.pi)

    def forward(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
    ) -> Tensor:

        head_re = self.node_emb(head_index)
        head_im = self.node_emb_im(head_index)
        tail_re = self.node_emb(tail_index)
        tail_im = self.node_emb_im(tail_index)

        rel_theta = self.rel_emb(rel_type)
        rel_re, rel_im = torch.cos(rel_theta), torch.sin(rel_theta)

        re_score = (rel_re * head_re - rel_im * head_im) - tail_re
        im_score = (rel_re * head_im + rel_im * head_re) - tail_im
        complex_score = torch.stack([re_score, im_score], dim=2)
        score = torch.linalg.vector_norm(complex_score, dim=(1, 2))

        return self.margin - score

    def loss(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
    ) -> Tensor:

        pos_score = self(head_index, rel_type, tail_index)
        neg_score = self(*self.random_sample(head_index, rel_type, tail_index))
        scores = torch.cat([pos_score, neg_score], dim=0)

        pos_target = torch.ones_like(pos_score)
        neg_target = torch.zeros_like(neg_score)
        target = torch.cat([pos_target, neg_target], dim=0)

        return F.binary_cross_entropy_with_logits(scores, target)
