import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Embedding

from torch_geometric.nn.kge import KGEModel


class TransD(KGEModel):
    r"""The TransD model from the `"Knowledge Graph Embedding via Dynamic
    Mapping Matrix" <https://aclanthology.org/P15-1067.pdf>`_ paper.

    :class:`TransD` projects head and tail to relation embedding space with
    dynamic mapping matrix constructed from node projection vector and relation
    projection vector as:

    .. math::
        \mathbf{h}_\perp = \mathbf{h}_p^T\mathbf{h}\mathbf{r}_p +
        \mathbf{I}\mathbf{h}

        \mathbf{t}_\perp = \mathbf{t}_p^T\mathbf{t}\mathbf{r}_p +
        \mathbf{I}\mathbf{t}

    It then models the relations as a translation from projected heads
    to projected tails in relation embedding space.

    .. math::
        \mathbf{h}_\perp + \mathbf{r} \approx \mathbf{t}_\perp,

    resulting in the scoring function:

    .. math::
        d(h, r, t) = - {\| \mathbf{h}_\perp + \mathbf{r} - \mathbf{t}_\perp
        \|}_p

    .. note::

        For an example of using the :class:`TransD` model, see
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

        self.node_proj_emb = Embedding(num_nodes, hidden_channels,
                                       sparse=sparse)
        self.rel_proj_emb = Embedding(num_relations, hidden_channels,
                                      sparse=sparse)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.node_emb.weight)
        torch.nn.init.xavier_uniform_(self.node_proj_emb.weight)
        torch.nn.init.xavier_uniform_(self.rel_emb.weight)
        torch.nn.init.xavier_uniform_(self.rel_proj_emb.weight)

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
        rel = F.normalize(rel, p=self.p_norm, dim=-1)
        tail = F.normalize(tail, p=self.p_norm, dim=-1)

        # Project head and tail to relation embedding space.
        head_proj = project_vector_ops(head, self.node_proj_emb(head_index),
                                       self.rel_proj_emb(rel_type))
        tail_proj = project_vector_ops(tail, self.node_proj_emb(tail_index),
                                       self.rel_proj_emb(rel_type))

        head_proj = F.normalize(head_proj, p=self.p_norm, dim=-1)
        tail_proj = F.normalize(tail_proj, p=self.p_norm, dim=-1)

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


def project_vector_ops(
    node_emb: Tensor,
    node_proj_emb: Tensor,
    rel_proj_emb: Tensor,
) -> Tensor:
    r"""Compute the projected vector of the node.

    .. math::
        \mathbf{v}_\perp = \mathbf{v}_p^T\mathbf{v}\mathbf{r}_p +
        \mathbf{I}\mathbf{v}

    Args:
        node_emb (Tensor): The node embedding.
        node_proj_emb (Tensor): The node projection embedding.
        rel_proj_emb (Tensor): The relation projection embedding.

    Returns:
        Tensor: The projected vector of node to relation embedding space.
    """
    out = (node_emb * node_proj_emb).sum(dim=-1).reshape(-1, 1)
    out = out * rel_proj_emb + node_emb
    return out
