import torch
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn.kge import KGEModel


class ComplEx(KGEModel):
    r"""The ComplEx model from the `"Complex Embeddings for Simple Link Prediction"
    <https://arxiv.org/pdf/1606.06357.pdf>

    :class:`ComplEx` models relations as complex-valued bilinear mappings between head
    and tail entities using the Hermetian dot product. The entities and relations are embedded
    in different dimensional spaces, resulting in the scoring function:

    .. math::
        d(h, r, t) = Re(< \mathbf{e}_h,  \mathbf{w}_r, \mathbf{e}_t>)

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
        super().__init__(num_nodes, num_relations, hidden_channels, sparse,
                         uses_complex=True)

    def forward(self, head_index: Tensor, rel_type: Tensor,
                tail_index: Tensor) -> Tensor:
        head_re = self.node_emb(head_index)
        head_im = self.node_emb_im(head_index)
        rel_re = self.rel_emb(rel_type)
        rel_im = self.rel_emb_im(rel_type)
        tail_re = self.node_emb(tail_index)
        tail_im = self.node_emb_im(tail_index)

        triple_dot = lambda x, y, z: torch.sum(x * y * z, dim=-1)
        return triple_dot(head_re, rel_re, tail_re) + \
            triple_dot(head_im, rel_re, tail_im) + \
            triple_dot(head_re, rel_im, tail_im) - \
            triple_dot(head_im, rel_im, tail_re)

    def loss(self, head_index: Tensor, rel_type: Tensor,
             tail_index: Tensor) -> Tensor:
        pos_score = self(head_index, rel_type, tail_index)
        neg_score = self(*self.random_sample(head_index, rel_type, tail_index))
        input = torch.cat((pos_score, neg_score))
        target = torch.cat((torch.ones_like(pos_score).long(),
                            -torch.ones_like(neg_score).long()))

        return F.nll_loss(F.logsigmoid(input), target)
