import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops

from ..inits import normal


class FeaStConv(MessagePassing):
    r"""The (translation-invariant) feature-steered convolutional operator from
    the `"FeaStNet: Feature-Steered Graph Convolutions for 3D Shape Analysis"
    <https://arxiv.org/abs/1706.05206>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \sum_{h=1}^H \frac{1}{|\mathcal{N}(i)|}
        \sum_{j \in \mathcal{N}(i)} q_h(\mathbf{x}_i, \mathbf{x}_j)
        \mathbf{W}_h \mathbf{x}_j

    with :math:`q_h(\mathbf{x}_i, \mathbf{x}_j) = \mathrm{softmax}_j
    (\mathbf{u}_h^{\top} (\mathbf{x}_j - \mathbf{x}_i) + c_h)`, where :math:`H`
    denotes the number of attention heads, and :math:`\mathbf{W}_h`,
    :math:`\mathbf{u}_h` and :math:`c_h` are trainable parameters.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of attention heads :math:`H`.
            (default: :obj:`1`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels, out_channels, heads=1, bias=True,
                 **kwargs):
        super(FeaStConv, self).__init__(aggr='mean', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads

        self.weight = Parameter(
            torch.Tensor(in_channels, heads * out_channels))
        self.u = Parameter(torch.Tensor(in_channels, heads))
        self.c = Parameter(torch.Tensor(heads))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        normal(self.weight, mean=0, std=0.1)
        normal(self.u, mean=0, std=0.1)
        normal(self.c, mean=0, std=0.1)
        normal(self.bias, mean=0, std=0.1)

    def forward(self, x, edge_index):
        """"""
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        q = torch.mm((x_j - x_i), self.u) + self.c  # Translation invariance.
        q = F.softmax(q, dim=1)

        x_j = torch.mm(x_j, self.weight).view(x_j.size(0), self.heads, -1)

        return x_j * q.view(-1, self.heads, 1)

    def update(self, aggr_out):
        aggr_out = aggr_out.sum(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias

        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
