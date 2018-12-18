import torch
from torch.nn import Parameter
from torch_geometric.utils import scatter_

from ..inits import reset, uniform


class NNConv(torch.nn.Module):
    r"""The continuous kernel-based convolutional operator adapted from the
    `"Neural Message Passing for Quantum Chemistry"
    <https://arxiv.org/abs/1704.01212>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta} \mathbf{x}_i +
        \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \cdot
        h_{\mathbf{\Theta}}(\mathbf{e}_{i,j}),

    where :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.*
    a MLP.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        nn (nn.Sequential): Neural network :math:`h_{\mathbf{\Theta}}`.
        aggr (string): The aggregation operator to use (one of :obj:`"add"`,
            :obj:`"mean"`, :obj:`"max"`). (default: :obj:`"add"`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add the transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 nn,
                 aggr="add",
                 root_weight=True,
                 bias=True):
        super(NNConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nn = nn
        self.aggr = aggr

        if root_weight:
            self.root = Parameter(torch.Tensor(in_channels, out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        size = self.in_channels
        uniform(size, self.root)
        uniform(size, self.bias)

    def forward(self, x, edge_index, pseudo):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        pseudo = pseudo.unsqueeze(-1) if pseudo.dim() == 1 else pseudo
        row, col = edge_index

        out = self.nn(pseudo)
        out = out.view(-1, self.in_channels, self.out_channels)
        out = torch.matmul(x[col].unsqueeze(1), out).squeeze(1)
        out = scatter_(self.aggr, out, row, dim_size=x.size(0))

        if self.root is not None:
            out = out + torch.mm(x, self.root)

        if self.bias is not None:
            out = out + self.bias

        return out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
