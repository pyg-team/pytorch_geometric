import torch
from torch import Tensor
from torch.nn import Linear, Parameter, ReLU, Sequential, Sigmoid

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj, OptTensor, SparseTensor
from torch_geometric.utils import spmm

from ..inits import glorot, zeros


class PDNConv(MessagePassing):
    r"""The pathfinder discovery network convolutional operator from the
    `"Pathfinder Discovery Networks for Neural Message Passing"
    <https://arxiv.org/pdf/2010.12878.pdf>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(v) \cup
        \{i\}}f_{\Theta}(\textbf{e}_{(j,i)}) \cdot f_{\Omega}(\mathbf{x}_{j})

    where :math:`z_{i,j}` denotes the edge feature vector from source node
    :math:`j` to target node :math:`i`, and :math:`\mathbf{x}_{j}` denotes the
    node feature vector of node :math:`j`.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        edge_dim (int): Edge feature dimensionality.
        hidden_channels (int): Hidden edge feature dimensionality.
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        normalize (bool, optional): Whether to add self-loops and compute
            symmetric normalization coefficients on the fly.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})`,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge features :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})`
    """
    def __init__(self, in_channels: int, out_channels: int, edge_dim: int,
                 hidden_channels: int, add_self_loops: bool = True,
                 normalize: bool = True, bias: bool = True, **kwargs):

        kwargs.setdefault("aggr", "add")
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.hidden_channels = hidden_channels
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self.lin = Linear(in_channels, out_channels, bias=False)

        self.mlp = Sequential(
            Linear(edge_dim, hidden_channels),
            ReLU(inplace=True),
            Linear(hidden_channels, 1),
            Sigmoid(),
        )

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        glorot(self.lin.weight)
        glorot(self.mlp[0].weight)
        glorot(self.mlp[2].weight)
        zeros(self.mlp[0].bias)
        zeros(self.mlp[2].bias)
        zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Adj,
                edge_attr: OptTensor = None) -> Tensor:
        """"""

        if isinstance(edge_index, SparseTensor):
            edge_attr = edge_index.storage.value()

        if edge_attr is not None:
            edge_attr = self.mlp(edge_attr).squeeze(-1)

        if isinstance(edge_index, SparseTensor):
            edge_index = edge_index.set_value(edge_attr, layout='coo')

        if self.normalize:
            if isinstance(edge_index, Tensor):
                edge_index, edge_attr = gcn_norm(edge_index, edge_attr,
                                                 x.size(self.node_dim), False,
                                                 self.add_self_loops,
                                                 self.flow, x.dtype)
            elif isinstance(edge_index, SparseTensor):
                edge_index = gcn_norm(edge_index, None, x.size(self.node_dim),
                                      False, self.add_self_loops, self.flow,
                                      x.dtype)

        x = self.lin(x)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_attr, size=None)

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')
