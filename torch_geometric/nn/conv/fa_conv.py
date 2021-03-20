from typing import Callable
from torch_geometric.typing import OptTensor

import torch
from torch import Tensor
from torch.nn import Linear, LeakyReLU
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm


class FAConv(MessagePassing):
    r"""Frequency Adaptive Graph Convolution operator from the `"Beyond Low-
    frequency Information in Graph Convolutional Networks"
    <https://arxiv.org/abs/2101.00797>`_ paper.

    .. math::
        \mathbf{x}\prime_0=\phi(W_1\mathbf{x}_0)

        \mathbf{x}\prime_i=W_2(\varepsilon \mathbf{x}\prime_0 +
        \sum_{j\epsilon\mathcal{N}_i} \frac{\alpha_{i,j}}{\sqrt{d_id_j}}
        \mathbf{x}_{j})

    Where :math:`\mathbf{x}_0` is the initial feature representation. and
    :math:`\phi` is an activation function.
    And the coefficients :math:`\alpha_{i,j}` are computed as.

    .. math::
        \mathbf{\alpha}_{i,j}=\textrm{tanh}(g^T[\mathbf{x}_i||\mathbf{x}_j])


    Args:
        in_channels (int or tuple): Size of each input sample.
        out_channels (int): Size of each output sample.
        raw_in_channels (int): Size of initial input sample :math:`x_0`.
        phi (torch.nn.Module, optinal): Activation function :math:`\phi`.
            (default: :obj:`LeakyReLU()`).
        epsilon (float, optional): A value between 0-1 which denotes strength
            of intial features :math:`\varepsilon`. (default: :obj:`0.1`).
        dropout (float, optional): Dropout probability of the normalized
            coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`).
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\frac{1}{\sqrt{d_id_j}}` on first
            execution, and will use the cached value for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    _alpha: OptTensor

    def __init__(self, in_channels: int, out_channels: int,
                 raw_in_channels: int, phi: Callable = LeakyReLU(),
                 epsilon: float = 0.1, dropout: float = 0.0,
                 add_self_loops: bool = False, cached: bool = False, **kwargs):

        assert epsilon <= 1 and epsilon >= 0
        super(FAConv, self).__init__(aggr='add', node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.raw_in_channels = raw_in_channels
        self.dropout = dropout
        self.cached = cached
        self.epsilon = torch.tensor(epsilon)
        self.add_self_loops = add_self_loops

        self.g_l = Linear(in_channels, 1, False)
        self.g_r = Linear(in_channels, 1, False)
        self.lin1 = Linear(raw_in_channels, in_channels, False)
        self.lin2 = Linear(in_channels, out_channels, False)
        self.phi = phi

        self._alpha = None
        self._cached_edge_index = None
        self.reset_parameters()

    def reset_parameters(self):
        self.g_l.reset_parameters()
        self.g_r.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self._alpha = None
        self._cached_edge_index = None

    def forward(self, x: Tensor, edge_index: Tensor,
                edge_weight: OptTensor = None, x_0: OptTensor = None,
                return_coeff: bool = False):
        r"""
        Args:
            return_coeff (bool): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, alpha)`, holding the normalized
                coefficient for each edge. (default: :obj:`False`)
        """
        cache = self._cached_edge_index
        if cache is None:
            edge_index, edge_weight = gcn_norm(
                edge_index, edge_weight, x.size(self.node_dim), dtype=x.dtype,
                add_self_loops=self.add_self_loops)
            if self.cached:
                self._cached_edge_index = (edge_index, edge_weight)
        else:
            edge_index, edge_weight = cache[0], cache[1]

        alpha_l = self.g_l(x)
        alpha_r = self.g_r(x)
        out = self.propagate(edge_index, x=x, alpha=(alpha_l, alpha_r),
                             edge_weight=edge_weight)

        out += self.epsilon * self.phi(
            self.lin1(x_0)) if x_0 is not None else 0
        out = self.lin2(out)
        alpha = self._alpha
        self._alpha = None

        if return_coeff:
            return out, (edge_index, alpha)
        else:
            return out

    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: Tensor,
                edge_weight: Tensor) -> Tensor:
        alpha = alpha_j + alpha_i
        alpha = torch.tanh(alpha) * edge_weight.unsqueeze(1)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha

    def __repr__(self):
        return '{}({}, {} , {}, phi={}, dropout={}, add_self_loops={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.raw_in_channels, self.phi, self.dropout, self.add_self_loops)
