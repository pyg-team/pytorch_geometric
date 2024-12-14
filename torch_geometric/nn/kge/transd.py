import math

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Embedding

from torch_geometric.nn.kge import KGEModel


class TransD(KGEModel):
    r"""The TransD model from the `"Knowledge Graph Embedding via Dynamic
    Mapping Matrix" <https://aclanthology.org/P15-1067.pdf>`_ paper.

    :class:`TransD` models relations as a translation from head to tail
    entities, after a mapping matrix has been applied, such that

    .. math::
        \mathbf{h}_\perp + \mathbf{r} \approx \mathbf{t}_perp,

    where
    .. math::
        \mathbf{h}_\perp = \mathbf{M}_{rh} h
        \mathbf{t}_\perp = \mathbf{M}_{th} t
        \mathbf{M}_{rh} = r_p h_p^\top + \mathbf{I}^{m \times n}
        \mathbf{M}_{rt} = r_p t_p^\top + \mathbf{I}^{m \times n}

    and :math:`h,  t, h_p, t_p \in \mathcal{R}^n` and
    :math:`r, r_p \in \mathcal{R}^m` are all learned embeddings.

    Similar to :obj:`TransE`, this results in the the scoring function:

    .. math::
        d(h_\perp, r, t_\perp) =
            - {\| \mathbf{h}_\perp + \mathbf{r} - \mathbf{t}_\perp \|}_p

    This score is optimized with the :obj:`margin_ranking_loss` by creating
    corrupted triplets. By default either the head or the tail of is corrupted
    uniformly at random. When :obj:`bern=True`, the head or tail is chosen
    proportional to the average number of heads per tail and tails per head for
    the relation, as described in the `"Knowledge Graph Embedding by
    Translating on Hyperplanes" <https://cdn.aaai.org/ojs/8870/
    8870-13-12398-1-2-20201228.pdf>`_ paper.

    .. note::

        For an example of using the :class:`TransD` model, see
        `examples/kge_fb15k_237.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        kge_fb15k_237.py>`_.

    Args:
        num_nodes (int): The number of nodes/entities in the graph.
        num_relations (int): The number of relations in the graph.
        hidden_channels_node (int): The node hidden embedding size (:math:`n`).
        hidden_channels_node (int): The relation hidden embedding size
            (:math:`m`).
        margin (int, optional): The margin of the ranking loss.
            (default: :obj:`1.0`)
        p_norm (int, optional): The order embedding and distance normalization.
            (default: :obj:`1.0`)
        sparse (bool, optional): If set to :obj:`True`, gradients w.r.t. to the
            embedding matrices will be sparse. (default: :obj:`False`)
        bern (bool, optional): If true, corrupt triplets using the Bernoulli
            strategy.  Otherwise, corrupt the head or tail uniformly.
            (default: :obj:`True`)
    """
    def __init__(self, num_nodes: int, num_relations: int,
                 hidden_channels_node: int, hidden_channels_rel: int,
                 margin: float = 1.0, p_norm: float = 1.0,
                 sparse: bool = False, bern: bool = False):
        super().__init__(num_nodes, num_relations, hidden_channels_node,
                         sparse, bern)
        self.hidden_channels_node = hidden_channels_node
        self.hidden_channels_rel = hidden_channels_rel

        # self.node_emb initialized in super
        self.rel_emb = Embedding(num_relations, hidden_channels_rel,
                                 sparse=sparse)
        self.rel_proj_emb = Embedding(num_relations, hidden_channels_rel,
                                      sparse=sparse)
        self.node_proj_emb = Embedding(num_nodes, hidden_channels_node,
                                       sparse=sparse)
        self.p_norm = p_norm
        self.margin = margin

        self.reset_parameters()

    def reset_parameters(self):
        bound = 6. / math.sqrt(self.hidden_channels_node)
        torch.nn.init.uniform_(self.node_emb.weight, -bound, bound)
        bound = 6. / math.sqrt(self.hidden_channels_rel)
        torch.nn.init.uniform_(self.rel_emb.weight, -bound, bound)
        # Achieve M_{rel node} initialized as Identity.
        torch.nn.init.zeros_(self.node_proj_emb.weight)
        torch.nn.init.zeros_(self.rel_proj_emb.weight)

    def _enforce_norm(self, emb):
        # Enforce norms in emb are <= 1.

        norm = emb.norm(p=self.p_norm, dim=-1, keepdim=True)
        mask = (norm > 1).view(-1)
        # We cannot modify emb in place as it's still needed for backprop.
        emb = emb.clone()
        emb[mask] = emb[mask] / norm[mask]
        return emb

    def _compute_perp(self, node_p, node, rel_p):
        # Compute the product:
        #   M_{rel node} node
        # As:
        #   (node_p^T node) rel_p + I node

        # (node^T node_p) rel_p, using efficient batch inner product.
        # TODO inner = (node * node_p).sum(dim=-1, keepdim=True)
        npt_n_rp = (node_p.unsqueeze(1) @ node.unsqueeze(2)).squeeze(2) * rel_p

        # Efficiently "multiply" non-square Identity with node.
        if self.hidden_channels_node >= self.hidden_channels_rel:
            id_node = node[:, self.hidden_channels_rel:]
        else:
            id_node = torch.zeros(node.size(0), self.hidden_channels_rel,
                                  device=node.device)
            id_node[:, :self.hidden_channels_node] = node

        return npt_n_rp + id_node

    def forward(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
    ) -> Tensor:

        # Lookup the embeddings.
        head = self.node_emb(head_index)
        head_proj = self.node_proj_emb(head_index)
        rel = self.rel_emb(rel_type)
        rel_proj = self.rel_proj_emb(rel_type)
        tail = self.node_emb(tail_index)
        tail_proj = self.node_proj_emb(tail_index)

        # Unlike TransE, which enforces head and tail norms == 1, TransD
        # enforces head, head_proj, tail, tail_proj, and rel norms <= 1.
        head = self._enforce_norm(head)
        head_proj = self._enforce_norm(head_proj)
        rel = self._enforce_norm(rel)
        tail = self._enforce_norm(tail)
        tail_proj = self._enforce_norm(tail_proj)

        # Compute the projected vectors, notated with \perp.
        head_perp = self._compute_perp(head_proj, head, rel_proj)
        tail_perp = self._compute_perp(tail_proj, tail, rel_proj)

        # Now the score is similar to TransE.
        return -(head_perp + rel - tail_perp).norm(p=self.p_norm, dim=-1)

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

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.num_nodes}, '
                f'num_relations={self.num_relations}, '
                f'hidden_channels_node={self.hidden_channels_node}, '
                f'hidden_channels_rel={self.hidden_channels_rel})')
