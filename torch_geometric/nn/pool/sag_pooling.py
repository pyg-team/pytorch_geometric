import torch
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_geometric.nn import GraphConv, GATConv, SAGEConv
import torch.nn.functional as F

from torch_geometric.nn.inits import uniform
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.nn.pool.topk_pool import topk, filter_adj


class SAGPooling(torch.nn.Module):
    r""":math:`\mathrm{top}_k` pooling operator from the `"Graph U-Net"
    <https://openreview.net/forum?id=HJePRoAct7>`_ and `"Towards Sparse
    Hierarchical Graph Classifiers" <https://arxiv.org/abs/1811.01287>`_ papers

    .. math::
        \mathbf{y} &= \frac{\mathbf{X}\mathbf{p}}{\| \mathbf{p} \|}
        \mathbf{i} &= \mathrm{top}_k(\mathbf{y})
        \mathbf{X}^{\prime} &= (\mathbf{X} \odot
        \mathrm{tanh}(\mathbf{y}))_{\mathbf{i}}
        \mathbf{A}^{\prime} &= \mathbf{A}_{\mathbf{i},\mathbf{i}},

    where nodes are dropped based on a learnable projection score
    :math:`\mathbf{p}`.


    Args:
        in_channels (int): Size of each input sample.
        ratio (float): Graph pooling ratio, which is used to compute
            :math:`k = \lceil \mathrm{ratio} \cdot N \rceil`.
            (default: :obj:`0.5`)
        gnn ('GCN' or 'GAT' or 'SAGE'): Specifies which GNN to use for calculating scores
            (default: :obj:`GCN`)
    """

    def __init__(self, in_channels, ratio=0.5, gnn='GCN', **kwargs):
        super(SAGPooling, self).__init__()

        self.in_channels = in_channels
        self.ratio = ratio

        self.weight = Parameter(torch.Tensor(1, self.in_channels))
        if gnn == 'GCN':
            self.gnn = GraphConv(in_channels=self.in_channels, out_channels=1, **kwargs)
        elif gnn =='GAT':
            self.gnn = GATConv(in_channels=self.in_channels, out_channels=1, **kwargs)
        elif gnn == 'SAGE':
            self.gnn = SAGEConv(in_channels=self.in_channels, out_channels=1, **kwargs)
        else:
            raise Exception("gnn must be 'GCN' or 'GAT' or 'SAGE' only. The value of gnn was: {}".format(gnn))

        self.reset_parameters()

    def reset_parameters(self):
        self.gnn.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """"""
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        x = x.unsqueeze(-1) if x.dim() == 1 else x

        score = self.gnn(x, edge_index).view(-1)
        perm = topk(score, self.ratio, batch)
        x = x[perm] * torch.tanh(score[perm]).view(-1, 1)
        batch = batch[perm]
        edge_index, edge_attr = filter_adj(
            edge_index, edge_attr, perm, num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm

    def __repr__(self):
        return '{}({}, ratio={})'.format(self.__class__.__name__,
                                         self.in_channels, self.ratio)
