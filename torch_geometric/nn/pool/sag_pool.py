import torch
from torch_geometric.nn import GraphConv, GATConv, SAGEConv
from torch_geometric.nn.pool.topk_pool import topk, filter_adj


class SAGPooling(torch.nn.Module):
    r"""The self-attention pooling operator from the `"Self-Attention Graph
    Pooling" <https://arxiv.org/abs/1904.08082>`_  paper

    .. math::
        \mathbf{y} &= \textrm{GNN}(\mathbf{X}, \mathbf{A})

        \mathbf{i} &= \mathrm{top}_k(\mathbf{y})

        \mathbf{X}^{\prime} &= (\mathbf{X} \odot
        \mathrm{tanh}(\mathbf{y}))_{\mathbf{i}}

        \mathbf{A}^{\prime} &= \mathbf{A}_{\mathbf{i},\mathbf{i}},

    where nodes are dropped based on a learnable projection score
    :math:`\mathbf{p}`.
    Projections scores are learned based on a graph neural network layer.

    Args:
        in_channels (int): Size of each input sample.
        ratio (float): Graph pooling ratio, which is used to compute
            :math:`k = \lceil \mathrm{ratio} \cdot N \rceil`.
            (default: :obj:`0.5`)
        gnn (string, optional): Specifies which graph neural network layer to
            use for calculating projection scores (one of
            :obj:`"GCN"`, :obj:`"GAT"` or :obj:`"SAGE"`). (default: :obj:`GCN`)
        **kwargs (optional): Additional parameters for initializing the graph
            neural network layer.
    """

    def __init__(self, in_channels, ratio=0.5, gnn='GCN', **kwargs):
        super(SAGPooling, self).__init__()

        self.in_channels = in_channels
        self.ratio = ratio
        self.gnn_name = gnn

        assert gnn in ['GCN', 'GAT', 'SAGE']
        if gnn == 'GCN':
            self.gnn = GraphConv(self.in_channels, 1, **kwargs)
        elif gnn == 'GAT':
            self.gnn = GATConv(self.in_channels, 1, **kwargs)
        else:
            self.gnn = SAGEConv(self.in_channels, 1, **kwargs)

        self.reset_parameters()

    def reset_parameters(self):
        self.gnn.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """"""
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        x = x.unsqueeze(-1) if x.dim() == 1 else x

        score = torch.tanh(self.gnn(x, edge_index).view(-1))
        perm = topk(score, self.ratio, batch)
        x = x[perm] * score[perm].view(-1, 1)
        batch = batch[perm]
        edge_index, edge_attr = filter_adj(
            edge_index, edge_attr, perm, num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm

    def __repr__(self):
        return '{}({}, {}, ratio={})'.format(self.__class__.__name__,
                                             self.gnn_name, self.in_channels,
                                             self.ratio)
