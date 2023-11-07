import math

import torch
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn.kge import KGEModel


class TransH(KGEModel):
    r"""The TransH model from the `"Knowledge Graph Embedding by Translating
    on Hyperplanes" <https://www.researchgate.net/publication/
    319207032_Knowledge_Graph_Embedding_by_Translating_on_Hyperplanes>`_ paper.

    :class:`TransH` models relations as a translation from projected head to tail
    entities such that 

    .. math::
        \mathbf{e}_{h_{\perp}} + \mathbf{d}_r \approx \mathbf{e}_{t_{\perp}},

    resulting in the scoring function:

    .. math::
        - || (\mathbf{e_h} - \mathbf{w}_r^T\mathbf{e_hw}_r) + \mathbf{d}_r - (\mathbf{e_t} - \mathbf{w}_r^T\mathbf{e_tw}_r) ||_2^2 
    .. note::

        For an example of using the :class:`TransH` model, see #TODO.
        # `examples/kge_fb15k_237.py
        # <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        # kge_fb15k_237.py>`_.

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

        # Calculate *negative* TransE norm:
        return -((head + rel) - tail).norm(p=self.p_norm, dim=-1)

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
