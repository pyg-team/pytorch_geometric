from typing import Optional, Tuple
from torch_geometric.typing import Adj, OptTensor, PairTensor

import torch
from torch import Tensor
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes

from ..inits import glorot, zeros

class PDNConv(MessagePassing):
    r"""The pathfinder discovery network convolutional operator from the
     `"Pathfinder Discovery Networks for Neural Message Passing"
    <https://arxiv.org/pdf/2010.12878.pdf>`_ paper.

    Its node-wise formulation is given by:

    .. math::
        \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(v) \cup
        \{ i \}} f_{Theta}(\textbf{z}_{(i,j)}) \cdot f_{\Theta^{'}}(\mathbf{x}_{j})

    , where :math:`z_{i,j}` denotes the edge feature vector from source node :obj:`j` to target
    node :obj:`i` and :math:`x_{j}` is the node feature vector of the source node :obj:`j`.

    Args:
        in_channels_node (int): Size of each input node feature sample.
        out_channels_node (int): Size of each output node feature sample.
        in_channels_edge (int): Size of each input edge feature sample.
        out_channels_edge (int): Size of each output edge feature sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        normalize (bool, optional): Whether to add self-loops and compute
            symmetric normalization coefficients on the fly.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, in_channels_node: int, out_channels_node: int,
    		  in_channels_edge: int, out_channels_edge: int,
                 improved: bool = False, add_self_loops: bool = True,
                 normalize: bool = True, bias: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super(PDNConv, self).__init__(**kwargs)

        self.in_channels_node = in_channels_node
        self.out_channels_node = out_channels_node
        self.in_channels_edge = in_channels_edge
        self.out_channels_edge = out_channels_edge
        self.improved = improved
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self.weight = Parameter(torch.Tensor(in_channels_node, out_channels_node))
        self.weight_e_0 = Parameter(torch.Tensor(in_channels_edge, out_channels_edge))
        self.weight_e_1 = Parameter(torch.Tensor(out_channels_edge, 1))
        self.bias_e_0 = Parameter(torch.Tensor(out_channels_edge))
        self.bias_e_1 = Parameter(torch.Tensor(1))


        if bias:
            self.bias = Parameter(torch.Tensor(out_channels_node))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.weight_e_0)
        glorot(self.weight_e_1)
        zeros(self.bias)
        zeros(self.bias_e_0)
        zeros(self.bias_e_1)

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: Tensor) -> Tensor:
        """"""
        
        edge_weight = edge_weight @ self.weight_e_0
        edge_weight = edge_weight + self.bias_e_0
        edge_weight = torch.relu(edge_weight)
        edge_weight = edge_weight @ self.weight_e_1
        edge_weight = edge_weight + self.bias_e_1
        edge_weight = torch.sigmoid(edge_weight)
        edge_weight = edge_weight.squeeze()   

        if self.normalize:
            edge_index, edge_weight = gcn_norm(
                edge_index, edge_weight, x.size(self.node_dim),
                self.improved, self.add_self_loops)
 

        x = x @ self.weight

        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {}, {}, {})'.format(self.__class__.__name__, self.in_channels_node,
                                   self.out_channels_node, self.in_channels_edge,
                                   self.out_channels_edge)
