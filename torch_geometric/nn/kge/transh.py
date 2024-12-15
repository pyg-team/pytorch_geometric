import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Embedding

from torch_geometric.nn.kge import KGEModel


class TransH(KGEModel):
    r"""The TransH model from the `"Knowledge Graph Embedding by Translating on
    Hyperplanes" <https://ojs.aaai.org/index.php/AAAI/article/view/8870>`_
    paper.

    :class:`TransH` models relations as translation from head to tail
    entities , where for each relation type, there is a hyperplane that the
    embeddings are projected onto. The scoring function is defined as:

    .. math::
        \mathbf{h}_r = \mathbf{h} - \mathbf{w}_r^T\mathbf{h}\mathbf{w}_r,
        \mathbf{t}_r = \mathbf{t} - \mathbf{w}_r^T\mathbf{t}\mathbf{w}_r


    resulting in the scoring function:

    .. math::
        d(h, r, t) = - {\| \mathbf{h}_r + \mathbf{r} - \mathbf{t}_r \|}_p


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

        # hyperplane embedding
        self.rel_normals = Embedding(num_relations, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):

        torch.nn.init.xavier_uniform_(self.node_emb.weight)
        torch.nn.init.xavier_uniform_(self.rel_emb.weight)
        torch.nn.init.xavier_uniform_(self.rel_normals.weight)

    def project(self, emb: Tensor, normal: Tensor) -> Tensor:
        '''
        Project entity embeddings onto the hyperplane defined by the
        relation type.
        '''
        norm = F.normalize(normal, p=2, dim=-1)
        return emb - (emb * norm).sum(dim=-1, keepdim=True) * norm

    def forward(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
    ) -> Tensor:

        head = self.node_emb(head_index)
        rel = self.rel_emb(rel_type)
        tail = self.node_emb(tail_index)

        head_proj = self.project(head, self.rel_normals(rel_type))
        tail_proj = self.project(tail, self.rel_normals(rel_type))

        head_proj = F.normalize(head_proj, p=self.p_norm, dim=-1)
        tail_proj = F.normalize(tail_proj, p=self.p_norm, dim=-1)

        # Calculate *negative* TransH norm:
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
