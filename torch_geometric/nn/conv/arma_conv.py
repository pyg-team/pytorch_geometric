from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Parameter, ReLU
from torch_sparse import matmul

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj, OptTensor, SparseTensor

from ..inits import glorot, zeros


class ARMAConv(MessagePassing):
    r"""The ARMA graph convolutional operator from the `"Graph Neural Networks
    with Convolutional ARMA Filters" <https://arxiv.org/abs/1901.01343>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \frac{1}{K} \sum_{k=1}^K \mathbf{X}_k^{(T)},

    with :math:`\mathbf{X}_k^{(T)}` being recursively defined by

    .. math::
        \mathbf{X}_k^{(t+1)} = \sigma \left( \mathbf{\hat{L}}
        \mathbf{X}_k^{(t)} \mathbf{W} + \mathbf{X}^{(0)} \mathbf{V} \right),

    where :math:`\mathbf{\hat{L}} = \mathbf{I} - \mathbf{L} = \mathbf{D}^{-1/2}
    \mathbf{A} \mathbf{D}^{-1/2}` denotes the
    modified Laplacian :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2}
    \mathbf{A} \mathbf{D}^{-1/2}`.

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        out_channels (int): Size of each output sample
            :math:`\mathbf{x}^{(t+1)}`.
        num_stacks (int, optional): Number of parallel stacks :math:`K`.
            (default: :obj:`1`).
        num_layers (int, optional): Number of layers :math:`T`.
            (default: :obj:`1`)
        act (callable, optional): Activation function :math:`\sigma`.
            (default: :meth:`torch.nn.ReLU()`)
        shared_weights (int, optional): If set to :obj:`True` the layers in
            each stack will share the same parameters. (default: :obj:`False`)
        dropout (float, optional): Dropout probability of the skip connection.
            (default: :obj:`0.`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})`,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge weights :math:`(|\mathcal{E}|)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})`
    """
    def __init__(self, in_channels: int, out_channels: int,
                 num_stacks: int = 1, num_layers: int = 1,
                 shared_weights: bool = False,
                 act: Optional[Callable] = ReLU(), dropout: float = 0.,
                 bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_stacks = num_stacks
        self.num_layers = num_layers
        self.act = act
        self.shared_weights = shared_weights
        self.dropout = dropout

        K, T, F_in, F_out = num_stacks, num_layers, in_channels, out_channels
        T = 1 if self.shared_weights else T

        self.weight = Parameter(torch.Tensor(max(1, T - 1), K, F_out, F_out))
        if in_channels > 0:
            self.init_weight = Parameter(torch.Tensor(K, F_in, F_out))
            self.root_weight = Parameter(torch.Tensor(T, K, F_in, F_out))
        else:
            self.init_weight = torch.nn.parameter.UninitializedParameter()
            self.root_weight = torch.nn.parameter.UninitializedParameter()
            self._hook = self.register_forward_pre_hook(
                self.initialize_parameters)

        if bias:
            self.bias = Parameter(torch.Tensor(T, K, 1, F_out))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        if not isinstance(self.init_weight, torch.nn.UninitializedParameter):
            glorot(self.init_weight)
            glorot(self.root_weight)
        zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""

        if isinstance(edge_index, Tensor):
            edge_index, edge_weight = gcn_norm(  # yapf: disable
                edge_index, edge_weight, x.size(self.node_dim),
                add_self_loops=False, flow=self.flow, dtype=x.dtype)

        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(  # yapf: disable
                edge_index, edge_weight, x.size(self.node_dim),
                add_self_loops=False, flow=self.flow, dtype=x.dtype)

        x = x.unsqueeze(-3)
        out = x
        for t in range(self.num_layers):
            if t == 0:
                out = out @ self.init_weight
            else:
                out = out @ self.weight[0 if self.shared_weights else t - 1]

            # propagate_type: (x: Tensor, edge_weight: OptTensor)
            out = self.propagate(edge_index, x=out, edge_weight=edge_weight,
                                 size=None)

            root = F.dropout(x, p=self.dropout, training=self.training)
            root = root @ self.root_weight[0 if self.shared_weights else t]
            out = out + root

            if self.bias is not None:
                out = out + self.bias[0 if self.shared_weights else t]

            if self.act is not None:
                out = self.act(out)

        return out.mean(dim=-3)

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    @torch.no_grad()
    def initialize_parameters(self, module, input):
        if isinstance(self.init_weight, nn.parameter.UninitializedParameter):
            F_in, F_out = input[0].size(-1), self.out_channels
            T, K = self.weight.size(0) + 1, self.weight.size(1)
            self.init_weight.materialize((K, F_in, F_out))
            self.root_weight.materialize((T, K, F_in, F_out))
            glorot(self.init_weight)
            glorot(self.root_weight)

        module._hook.remove()
        delattr(module, '_hook')

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, num_stacks={self.num_stacks}, '
                f'num_layers={self.num_layers})')
