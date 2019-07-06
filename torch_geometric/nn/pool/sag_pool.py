import torch
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GraphConv
from torch_geometric.nn.pool.topk_pool import topk, filter_adj
from torch_geometric.utils import softmax


class SAGPooling(torch.nn.Module):
    r"""The self-attention pooling operator from the `"Self-Attention Graph
    Pooling" <https://arxiv.org/abs/1904.08082>`_ and `"Understanding
    Attention and Generalization in Graph Neural Networks"
    <https://arxiv.org/abs/1905.02850>`_ papers

    if min_score :math:`\tilde{\alpha}` is None:

        .. math::
            \mathbf{y} &= \textrm{GNN}(\mathbf{X}, \mathbf{A})

            \mathbf{i} &= \mathrm{top}_k(\mathbf{y})

            \mathbf{X}^{\prime} &= (\mathbf{X} \odot
            \mathrm{tanh}(\mathbf{y}))_{\mathbf{i}}

            \mathbf{A}^{\prime} &= \mathbf{A}_{\mathbf{i},\mathbf{i}}

    if min_score :math:`\tilde{\alpha}` is a value in [0, 1]:

        .. math::
            \mathbf{y} &= \mathrm{softmax}(\textrm{GNN}(\mathbf{X},\mathbf{A}))

            \mathbf{i} &= \mathbf{y}_i > \tilde{\alpha}

            \mathbf{X}^{\prime} &= (\mathbf{X} \odot \mathbf{y})_{\mathbf{i}}

            \mathbf{A}^{\prime} &= \mathbf{A}_{\mathbf{i},\mathbf{i}},

    where nodes are dropped based on a learnable projection score
    :math:`\mathbf{p}`.
    Projections scores are learned based on a graph neural network layer.

    Args:
        in_channels (int): Size of each input sample.
        ratio (float): Graph pooling ratio, which is used to compute
            :math:`k = \lceil \mathrm{ratio} \cdot N \rceil`.
            This value is ignored if min_score is not None.
            (default: :obj:`0.5`)
        gnn (string, optional): Specifies which graph neural network layer to
            use for calculating projection scores (one of
            :obj:`"GCN"`, :obj:`"GAT"` or :obj:`"SAGE"`). (default: :obj:`GCN`)
        min_score (float, optional): Minimal node score :math:`\tilde{\alpha}`
            which is used to compute indices of pooled nodes
            :math:`\mathbf{i} = \mathbf{y}_i > \tilde{\alpha}`.
            When this value is not :obj:`None`, the :obj:`ratio` argument is
            ignored. (default: :obj:`None`)
        multiplier (float, optional): Coefficient by which features gets
            multiplied after pooling. This can be useful for large graphs and
            when :obj:`min_score` is used. (default: :obj:`1`)
        **kwargs (optional): Additional parameters for initializing the graph
            neural network layer.
    """

    def __init__(self, in_channels, ratio=0.5, gnn='GraphConv', min_score=None,
                 multiplier=1, **kwargs):
        super(SAGPooling, self).__init__()

        self.in_channels = in_channels
        self.ratio = ratio
        self.min_score = min_score
        self.multiplier = multiplier
        self.gnn_name = gnn

        assert gnn in ['GraphConv', 'GCN', 'GAT', 'SAGE']
        if gnn == 'GCN':
            self.gnn = GCNConv(self.in_channels, 1, **kwargs)
        elif gnn == 'GAT':
            self.gnn = GATConv(self.in_channels, 1, **kwargs)
        elif gnn == 'SAGE':
            self.gnn = SAGEConv(self.in_channels, 1, **kwargs)
        else:
            self.gnn = GraphConv(self.in_channels, 1, **kwargs)

        self.reset_parameters()

    def reset_parameters(self):
        self.gnn.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None, batch=None,
                attn_input=None):  # `attn_input` can differ from x in general.
        """"""
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        attn_input = x if attn_input is None else attn_input
        if attn_input.dim() == 1:
            attn_input = attn_input.unsqueeze(-1)

        score = self.gnn(attn_input, edge_index).view(-1)

        if self.min_score is None:
            score = torch.tanh(score)
        else:
            score = softmax(score, batch)

        perm = topk(score, self.ratio, batch, self.min_score)
        x = x[perm] * score[perm].view(-1, 1)
        x = self.multiplier * x if self.multiplier != 1 else x

        batch = batch[perm]
        edge_index, edge_attr = filter_adj(edge_index, edge_attr, perm,
                                           num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm, score[perm]

    def __repr__(self):
        return '{}({}, {}, {}={}, multiplier={})'.format(
            self.__class__.__name__, self.gnn_name, self.in_channels,
            'ratio' if self.min_score is None else 'min_score',
            self.ratio if self.min_score is None else self.min_score,
            self.multiplier)
