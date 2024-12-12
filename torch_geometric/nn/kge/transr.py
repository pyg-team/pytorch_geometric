import math

import torch
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn.kge import KGEModel


class TransR(KGEModel):
    r"""The TransR model from the `"Learning Entity and Relation Embeddings 
    for Knowledge Graph Completion" 
    <https://cdn.aaai.org/ojs/9491/9491-13-13019-1-2-20201228.pdf>`_ paper.

    :class:`TransR` models relations by first projecting entities from entity 
    space to relation space using relation-specific matrices, and then 
    performing translation in the relation space:

    .. math::
        \mathbf{h}_r &= \mathbf{h}M_r \\
        \mathbf{t}_r &= \mathbf{t}M_r \\
        \mathbf{h}_r + \mathbf{r} &\approx \mathbf{t}_r

    resulting in the scoring function:

    .. math::
        d(h, r, t) = - {\| \mathbf{h}M_r + \mathbf{r} - \mathbf{t}M_r \|}_p

    .. note::

        For an example of using the :class:`TransR` model, see
        `examples/kge_fb15k_237.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        kge_fb15k_237.py>`_.

    Args:
        num_nodes (int): The number of nodes/entities in the graph.
        num_relations (int): The number of relations in the graph.
        hidden_channels (int): The hidden embedding size.
        margin (float, optional): The margin of the ranking loss.
            (default: :obj:`1.0`)
        p_norm (float, optional): The order of norm used for embeddings 
        and distance.
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

        # Create relation-specific projection matrices
        self.rel_proj = torch.nn.Parameter(
            torch.Tensor(num_relations, hidden_channels, hidden_channels))

        self.reset_parameters()

    def reset_parameters(self):
        6. / math.sqrt(self.hidden_channels)

        torch.nn.init.xavier_uniform_(self.node_emb.weight)
        torch.nn.init.xavier_uniform_(self.rel_emb.weight)
        torch.nn.init.xavier_uniform_(self.rel_proj)

        F.normalize(self.rel_emb.weight.data, p=self.p_norm, dim=-1,
                    out=self.rel_emb.weight.data)

    def _project(self, entities: Tensor, rel_type: Tensor) -> Tensor:
        """
        Project entities to relation space using relation-specific matrices.
        """
        proj_matrices = self.rel_proj[rel_type]
        entities = entities.unsqueeze(1)
        projected = torch.bmm(entities, proj_matrices)
        return projected.squeeze(1)

    def forward(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
    ) -> Tensor:
        """Compute TransR scores for the given triplets."""
        head = self.node_emb(head_index)
        rel = self.rel_emb(rel_type)
        tail = self.node_emb(tail_index)

        head_proj = self._project(head, rel_type)
        tail_proj = self._project(tail, rel_type)

        head_proj = F.normalize(head_proj, p=self.p_norm, dim=-1)
        tail_proj = F.normalize(tail_proj, p=self.p_norm, dim=-1)

        # Calculate *negative* TransR norm:
        return -((head_proj + rel) - tail_proj).norm(p=self.p_norm, dim=-1)

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
