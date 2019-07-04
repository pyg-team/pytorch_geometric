import torch
from torch.nn import Parameter
from torch_scatter import scatter_add, scatter_max

from ..inits import uniform
from ...utils.num_nodes import maybe_num_nodes
from ...utils import softmax


def topk(x, ratio, batch, min_score=None, tol=1e-7):
    if min_score is not None:
        scores_max = scatter_max(x, batch)[0][batch] - tol  # make sure that we do not drop all nodes in a graph
        scores_min = torch.min(scores_max, x.new_full((1,), min_score))

        perm = torch.nonzero(x > scores_min).view(-1)

    else:
        num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0)
        batch_size, max_num_nodes = num_nodes.size(0), num_nodes.max().item()

        cum_num_nodes = torch.cat(
            [num_nodes.new_zeros(1),
             num_nodes.cumsum(dim=0)[:-1]], dim=0)

        index = torch.arange(batch.size(0), dtype=torch.long, device=x.device)
        index = (index - cum_num_nodes[batch]) + (batch * max_num_nodes)

        dense_x = x.new_full((batch_size * max_num_nodes,), -2)
        dense_x[index] = x
        dense_x = dense_x.view(batch_size, max_num_nodes)

        _, perm = dense_x.sort(dim=-1, descending=True)

        perm = perm + cum_num_nodes.view(-1, 1)
        perm = perm.view(-1)

        k = (ratio * num_nodes.to(torch.float)).ceil().to(torch.long)
        mask = [
            torch.arange(k[i], dtype=torch.long, device=x.device) +
            i * max_num_nodes for i in range(batch_size)
        ]
        mask = torch.cat(mask, dim=0)

        perm = perm[mask]

    return perm


def filter_adj(edge_index, edge_attr, perm, num_nodes=None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    mask = perm.new_full((num_nodes, ), -1)
    i = torch.arange(perm.size(0), dtype=torch.long, device=perm.device)
    mask[perm] = i

    row, col = edge_index
    row, col = mask[row], mask[col]
    mask = (row >= 0) & (col >= 0)
    row, col = row[mask], col[mask]

    if edge_attr is not None:
        edge_attr = edge_attr[mask]

    return torch.stack([row, col], dim=0), edge_attr


class TopKPooling(torch.nn.Module):
    r""":math:`\mathrm{top}_k` pooling operator from the `"Graph U-Net"
    <https://openreview.net/forum?id=HJePRoAct7>`_, `"Towards Sparse
    Hierarchical Graph Classifiers" <https://arxiv.org/abs/1811.01287>`_
    and `"Understanding Attention and Generalization in Graph Neural
    Networks" <https://arxiv.org/abs/1905.02850>`_ papers

    if min_score :math:`\tilde{\alpha}` is None:

        .. math::
            \mathbf{y} &= \frac{\mathbf{X}\mathbf{p}}{\| \mathbf{p} \|}

            \mathbf{i} &= \mathrm{top}_k(\mathbf{y})

            \mathbf{X}^{\prime} &= (\mathbf{X} \odot
            \mathrm{tanh}(\mathbf{y}))_{\mathbf{i}}

            \mathbf{A}^{\prime} &= \mathbf{A}_{\mathbf{i},\mathbf{i}}

    if min_score :math:`\tilde{\alpha}` is a value in [0, 1]:

        .. math::
            \mathbf{y} &= softmax(\mathbf{X}\mathbf{p})

            \mathbf{i} &: \mathbf{y}_i > \tilde{\alpha}

            \mathbf{X}^{\prime} &= (\mathbf{X} \odot (\mathbf{y}))_{\mathbf{i}}

            \mathbf{A}^{\prime} &= \mathbf{A}_{\mathbf{i},\mathbf{i}},

    where nodes are dropped based on a learnable projection score
    :math:`\mathbf{p}`.

    Args:
        in_channels (int): Size of each input sample.
        ratio (float): Graph pooling ratio, which is used to compute
            :math:`k = \lceil \mathrm{ratio} \cdot N \rceil`.
            This value is ignored if min_score is not None.
            (default: :obj:`0.5`)
        min_score (float): Minimal node score, which is used to compute indices of
            pooled nodes :math:`\mathbf{i} : \mathbf{y}_i > \tilde{\alpha}`.
            When this value is not None, the ratio argument is ignored.
            (default: None)
        multiplier (float): Coefficient by which multiply features after pooling.
            This can be useful for large graphs and when min_score is used.
            (default: None)
    """

    def __init__(self, in_channels, ratio=0.5, min_score=None, multiplier=None):
        super(TopKPooling, self).__init__()

        self.in_channels = in_channels
        self.ratio = ratio
        self.min_score = min_score
        self.multiplier = multiplier

        self.weight = Parameter(torch.Tensor(1, in_channels))

        self.reset_parameters()

    def reset_parameters(self):
        size = self.in_channels
        uniform(size, self.weight)

    def forward(self, x, edge_index, edge_attr=None, batch=None, attn_input=None):
        """"""
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        if attn_input is None:
            attn_input = x
        attn_input = attn_input.unsqueeze(-1) if attn_input.dim() == 1 else attn_input

        score = (attn_input * self.weight).sum(dim=-1)

        if self.min_score is None:
            score = torch.tanh(score / self.weight.norm(p=2, dim=-1))
        else:
            score = softmax(score, batch)

        perm = topk(score, self.ratio, batch, min_score=self.min_score)

        x = x[perm] * score[perm].view(-1, 1)
        if self.multiplier is not None:
            x = x * self.multiplier

        batch = batch[perm]
        edge_index, edge_attr = filter_adj(
            edge_index, edge_attr, perm, num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm, score[perm]

    def __repr__(self):
        return '{}({}, ratio={}{}, min_score={}, multiplier={})'.format(self.__class__.__name__,
                                                                        self.in_channels, self.ratio,
                                                                        '(ignored)' if self.min_score
                                                                                       is not None else '',
                                                                        self.min_score, self.multiplier)
