from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Embedding

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import HeteroLinear, Linear
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import softmax


class HEATConv(MessagePassing):
    r"""The heterogeneous edge-enhanced graph attentional operator from the
    `"Heterogeneous Edge-Enhanced Graph Attention Network For Multi-Agent
    Trajectory Prediction" <https://arxiv.org/abs/2106.07161>`_ paper, which
    enhances :class:`~torch_geometric.nn.conv.GATConv` by:

    1. type-specific transformations of nodes of different types
    2. edge type and edge feature incorporation, in which edges are assumed to
       have different types but contain the same kind of attributes

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        out_channels (int): Size of each output sample.
        num_node_types (int): The number of node types.
        num_edge_types (int): The number of edge types.
        edge_type_emb_dim (int): The embedding size of edge types.
        edge_dim (int): Edge feature dimensionality.
        edge_attr_emb_dim (int): The embedding size of edge features.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})`,
          edge indices :math:`(2, |\mathcal{E}|)`,
          node types :math:`(|\mathcal{V}|)`,
          edge types :math:`(|\mathcal{E}|)`,
          edge features :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})`
    """
    def __init__(self, in_channels: int, out_channels: int,
                 num_node_types: int, num_edge_types: int,
                 edge_type_emb_dim: int, edge_dim: int, edge_attr_emb_dim: int,
                 heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.0,
                 root_weight: bool = True, bias: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.root_weight = root_weight

        self.hetero_lin = HeteroLinear(in_channels, out_channels,
                                       num_node_types, bias=bias)

        self.edge_type_emb = Embedding(num_edge_types, edge_type_emb_dim)
        self.edge_attr_emb = Linear(edge_dim, edge_attr_emb_dim, bias=False)

        self.att = Linear(
            2 * out_channels + edge_type_emb_dim + edge_attr_emb_dim,
            self.heads, bias=False)

        self.lin = Linear(out_channels + edge_attr_emb_dim, out_channels,
                          bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.hetero_lin.reset_parameters()
        self.edge_type_emb.reset_parameters()
        self.edge_attr_emb.reset_parameters()
        self.att.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj, node_type: Tensor,
                edge_type: Tensor, edge_attr: OptTensor = None) -> Tensor:
        """"""
        x = self.hetero_lin(x, node_type)

        edge_type_emb = F.leaky_relu(self.edge_type_emb(edge_type),
                                     self.negative_slope)

        # propagate_type: (x: Tensor, edge_type_emb: Tensor, edge_attr: OptTensor)  # noqa
        out = self.propagate(edge_index, x=x, edge_type_emb=edge_type_emb,
                             edge_attr=edge_attr, size=None)

        if self.concat:
            if self.root_weight:
                out += x.view(-1, 1, self.out_channels)
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
            if self.root_weight:
                out += x

        return out

    def message(self, x_i: Tensor, x_j: Tensor, edge_type_emb: Tensor,
                edge_attr: Tensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        edge_attr = F.leaky_relu(self.edge_attr_emb(edge_attr),
                                 self.negative_slope)

        alpha = torch.cat([x_i, x_j, edge_type_emb, edge_attr], dim=-1)
        alpha = F.leaky_relu(self.att(alpha), self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = self.lin(torch.cat([x_j, edge_attr], dim=-1)).unsqueeze(-2)
        return out * alpha.unsqueeze(-1)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')
