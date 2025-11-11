import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Embedding

from torch_geometric.nn.kge import KGEModel


class SimplE(KGEModel):
    r"""The SimplE model from the `"SimplE Embedding for Link Prediction in
    Knowledge Graphs" <https://proceedings.neurips.cc/paper/2018/file/
    b2ab001909a8a6f04b51920306046ce5-Paper.pdf>`_ paper.

    :class:`SimplE` addresses the independence of the two embedding vectors
    for each entity in CP decomposition by using the inverse of relations.
    The scoring function for a triple :math:`(h, r, t)` is defined as:

    .. math::
        d(h, r, t) = \frac{1}{2}(\langle \mathbf{e}_h, \mathbf{v}_r,
        \mathbf{e}_t \rangle + \langle \mathbf{e}_t, \mathbf{v}_{r^{-1}},
        \mathbf{e}_h \rangle)

    where :math:`\langle \cdot, \cdot, \cdot \rangle` denotes the element-wise
    product followed by sum, and :math:`\mathbf{v}_{r^{-1}}` is the embedding
    for the inverse relation.

    .. note::

        For an example of using the :class:`SimplE` model, see
        `examples/contrib/simple_fb15k_237.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        contrib/simple_fb15k_237.py>`_.

    Args:
        num_nodes (int): The number of nodes/entities in the graph.
        num_relations (int): The number of relations in the graph.
        hidden_channels (int): The hidden embedding size.
        sparse (bool, optional): If set to :obj:`True`, gradients w.r.t. to
            the embedding matrices will be sparse. (default: :obj:`False`)
    """
    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        hidden_channels: int,
        sparse: bool = False,
    ):
        super().__init__(num_nodes, num_relations, hidden_channels, sparse)

        # Additional embeddings for tail entities and inverse relations
        self.node_emb_tail = Embedding(num_nodes, hidden_channels, sparse=sparse)
        self.rel_emb_inv = Embedding(num_relations, hidden_channels, sparse=sparse)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.node_emb.weight)
        torch.nn.init.xavier_uniform_(self.node_emb_tail.weight)
        torch.nn.init.xavier_uniform_(self.rel_emb.weight)
        torch.nn.init.xavier_uniform_(self.rel_emb_inv.weight)

    def forward(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
    ) -> Tensor:

        # Get embeddings
        head = self.node_emb(head_index)  # h_{e_i}
        tail = self.node_emb_tail(tail_index)  # t_{e_j}
        rel = self.rel_emb(rel_type)  # v_r
        rel_inv = self.rel_emb_inv(rel_type)  # v_{r^{-1}}

        # Get tail entity head embedding and head entity tail embedding
        # for the inverse part
        tail_head = self.node_emb(tail_index)  # h_{e_j}
        head_tail = self.node_emb_tail(head_index)  # t_{e_i}

        # Compute the two CP scores
        # Score 1: ⟨h_{e_i}, v_r, t_{e_j}⟩
        score1 = (head * rel * tail).sum(dim=-1)

        # Score 2: ⟨h_{e_j}, v_{r^{-1}}, t_{e_i}⟩
        score2 = (tail_head * rel_inv * head_tail).sum(dim=-1)

        # SimplE score is the average of the two
        return 0.5 * (score1 + score2)

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
