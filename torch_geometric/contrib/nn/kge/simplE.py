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
        r"""Initializes the SimplE model.
        
        SimplE extends CP decomposition by introducing inverse relations to
        couple the head and tail embeddings of entities. Each entity has two
        embeddings: one for when it appears as a head and one for when it
        appears as a tail. Similarly, each relation has two embeddings: one
        for the forward direction and one for the inverse direction.
        
        Args:
            num_nodes (int): The number of entities in the knowledge graph.
            num_relations (int): The number of relation types in the knowledge
                graph.
            hidden_channels (int): The dimensionality of the embedding
                vectors. Larger values can capture more complex patterns
                but require more memory and computation.
            sparse (bool, optional): If set to :obj:`True`, gradients w.r.t. to
                the embedding matrices will be sparse. Useful for very large
                knowledge graphs. (default: :obj:`False`)
        """
        super().__init__(num_nodes, num_relations, hidden_channels, sparse)

        # Additional embeddings beyond the base KGEModel:
        # - node_emb_tail: tail embeddings for entities (used when
        #   entity is a tail)
        # - rel_emb_inv: inverse relation embeddings (for r^{-1})
        self.node_emb_tail = Embedding(num_nodes, hidden_channels, sparse=sparse)
        self.rel_emb_inv = Embedding(num_relations, hidden_channels, sparse=sparse)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module.
        
        Initializes all embedding matrices using Xavier uniform initialization,
        which helps maintain the variance of activations and gradients through
        the network layers.
        """
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
        r"""Computes the score for the given triplet.
        
        The SimplE scoring function computes the average of two CP
        decomposition scores: one for the forward relation and one for
        the inverse relation. This addresses the independence issue in CP
        by coupling the head and tail embeddings of entities through
        inverse relations.
        
        Args:
            head_index (torch.Tensor): The head entity indices of shape
                :obj:`[batch_size]`.
            rel_type (torch.Tensor): The relation type indices of shape
                :obj:`[batch_size]`.
            tail_index (torch.Tensor): The tail entity indices of shape
                :obj:`[batch_size]`.
        
        Returns:
            torch.Tensor: The score for each triplet of shape
                :obj:`[batch_size]`. Higher scores indicate more
                plausible triples.
        """
        # Get embeddings for the forward direction: (h, r, t)
        head = self.node_emb(head_index)  # h_{e_i}: head emb of head entity
        tail = self.node_emb_tail(tail_index)  # t_{e_j}: tail emb of tail
        rel = self.rel_emb(rel_type)  # v_r: forward relation embedding

        # Get embeddings for the inverse direction: (t, r^{-1}, h)
        # For the inverse, we need the head embedding of the tail entity
        # and the tail embedding of the head entity
        tail_head = self.node_emb(tail_index)  # h_{e_j}: head emb of tail
        head_tail = self.node_emb_tail(head_index)  # t_{e_i}: tail emb
        rel_inv = self.rel_emb_inv(rel_type)  # v_{r^{-1}}: inverse relation

        # Compute Score 1: CP score for forward relation
        # ⟨h_{e_i}, v_r, t_{e_j}⟩ = sum over dimensions of (h * v_r * t)
        score1 = (head * rel * tail).sum(dim=-1)

        # Compute Score 2: CP score for inverse relation
        # ⟨h_{e_j}, v_{r^{-1}}, t_{e_i}⟩ = sum over dims of
        # (h_tail * v_r_inv * t_head)
        score2 = (tail_head * rel_inv * head_tail).sum(dim=-1)

        # SimplE score is the average of the two CP scores
        # This coupling ensures that both directions contribute to
        # learning
        return 0.5 * (score1 + score2)

    def loss(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
    ) -> Tensor:
        r"""Computes the loss for the given positive triplets.
        
        The loss function uses binary cross-entropy with logits, comparing
        positive triplets against randomly sampled negative triplets. This
        encourages the model to assign higher scores to positive triples than
        to negative ones.
        
        Args:
            head_index (torch.Tensor): The head entity indices of shape
                :obj:`[batch_size]`.
            rel_type (torch.Tensor): The relation type indices of shape
                :obj:`[batch_size]`.
            tail_index (torch.Tensor): The tail entity indices of shape
                :obj:`[batch_size]`.
        
        Returns:
            torch.Tensor: The computed loss value (a scalar).
        """
        # Compute scores for positive triplets
        pos_score = self(head_index, rel_type, tail_index)
        
        # Generate negative triplets by randomly corrupting heads or tails
        # and compute their scores
        neg_score = self(*self.random_sample(head_index, rel_type, tail_index))
        
        # Concatenate positive and negative scores
        scores = torch.cat([pos_score, neg_score], dim=0)

        # Create targets: 1 for positive, 0 for negative
        pos_target = torch.ones_like(pos_score)
        neg_target = torch.zeros_like(neg_score)
        target = torch.cat([pos_target, neg_target], dim=0)

        # Binary cross-entropy loss encourages positive scores to be high
        # and negative scores to be low
        return F.binary_cross_entropy_with_logits(scores, target)

