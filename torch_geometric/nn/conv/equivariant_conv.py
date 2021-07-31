from typing import Optional, Callable, Tuple
from torch_geometric.typing import OptTensor, Adj

import torch
from torch import Tensor
from torch_sparse import SparseTensor, set_diag
from torch_scatter import scatter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops

from ..inits import reset


class EquivariantConv(MessagePassing):
    r"""The Equivariant graph neural network operator form the
    `"E(n) Equivariant Graph Neural Networks"
    <https://arxiv.org/pdf/2102.09844.pdf>`_ paper

    .. math::
        \mathbf{m}_{ij}=h_{\mathbf{\Theta}}(\mathbf{x}_i,\mathbf{x}_j,\|
        {\mathbf{pos}_i-\mathbf{pos}_j}\|^2_2,\mathbf{e}_{ij})

        \mathbf{x}^{\prime}_i = \gamma_{\mathbf{\Theta}}(\mathbf{x}_i,
        \sum_{j \in \mathcal{N}(i)} \mathbf{m}_{ij})

        \mathbf{vel}^{\prime}_i = \phi_{\mathbf{\Theta}}(\mathbf{x}_i)\mathbf
        {vel}_i + \frac{1}{|\mathcal{N}(i)|}\sum_{j \in\mathcal{N}(i)}
        (\mathbf{pos}_i-\mathbf{pos}_j)
        \rho_{\mathbf{\Theta}}(\mathbf{m}_{ij})

        \mathbf{pos}^{\prime}_i = \mathbf{pos}_i + \mathbf{vel}_i

    where :math:`\gamma_{\mathbf{\Theta}}`,
    :math:`h_{\mathbf{\Theta}}`, :math:`\rho_{\mathbf{\Theta}}`
    and :math:`\phi_{\mathbf{\Theta}}` denote neural
    networks, *.i.e.* MLPs. :math:`\mathbf{P} \in \mathbb{R}^{N \times D}`
    and :math:`\mathbf{V} \in \mathbb{R}^{N \times D}`
    defines the position and velocity of each point respectively.

    Args:
        local_nn (torch.nn.Module, optional): A neural network
            :math:`h_{\mathbf{\Theta}}` that maps node features :obj:`x`,
            sqaured distance :math:`\|{\mathbf{pos}_i-\mathbf{pos}_j}\|^2_2`
            and edge_features :obj:`edge_attr`
            of shape :obj:`[-1, 2*in_channels + 1 +edge_dim]`
            to shape :obj:`[-1, hidden_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`. (default: :obj:`None`)
        pos_nn (torch.nn.Module,optinal): A neural network
            :math:`\rho_{\mathbf{\Theta}}` that
            maps message :obj:`m` of shape
            :obj:`[-1, hidden_channels]`,
            to shape :obj:`[-1, 1]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`. (default: :obj:`None`)
        vel_nn (torch.nn.Module,optional): A neural network
            :math:`\phi_{\mathbf{\Theta}}` that
            maps node featues :obj:`x` of shape :obj:`[-1, in_channels]`,
            to shape :obj:`[-1, 1]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`. (default: :obj:`None`)
        global_nn (torch.nn.Module, optional): A neural network
            :math:`\gamma_{\mathbf{\Theta}}` that maps
            message :obj:`m` after aggregation
            and node features :obj:`x` of shape
            :obj:`[-1, hidden_channels + in_channels]`
            to shape :obj:`[-1, out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`. (default: :obj:`None`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        aggr (string, optional): The operator used to aggregate message
            :obj:`m` (:obj:`"add"`, :obj:`"mean"`).
            (default: :obj:`"mean"`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, local_nn: Optional[Callable] = None,
                 pos_nn: Optional[Callable] = None,
                 vel_nn: Optional[Callable] = None,
                 global_nn: Optional[Callable] = None,
                 add_self_loops: bool = True, aggr="mean", **kwargs):
        super(EquivariantConv, self).__init__(aggr=aggr, **kwargs)

        self.local_nn = local_nn
        self.pos_nn = pos_nn
        self.vel_nn = vel_nn
        self.global_nn = global_nn
        self.add_self_loops = add_self_loops

        self.reset_parameters()

    def reset_parameters(self):

        reset(self.local_nn)
        reset(self.pos_nn)
        reset(self.vel_nn)
        reset(self.global_nn)

    def forward(self, x: OptTensor,
                pos: Tensor,
                edge_index: Adj, vel: OptTensor = None,
                edge_attr: OptTensor = None
                ) -> Tuple[Tensor, Tuple[Tensor, OptTensor]]:
        """"""
        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index,
                                               num_nodes=pos.size(0))
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        # propagate_type: (x: OptTensor, pos: Tensor, edge_attr: OptTensor) -> Tuple[Tensor,Tensor] # noqa
        out_x, out_pos = self.propagate(edge_index, x=x, pos=pos,
                                        edge_attr=edge_attr, size=None)

        out_x = out_x if x is None else torch.cat([x, out_x], dim=1)
        if self.global_nn is not None:
            out_x = self.global_nn(out_x)

        if vel is None:
            out_pos += pos
            out_vel = None
        else:
            out_vel = (vel if self.vel_nn is None or x is None else
                       self.vel_nn(x) * vel) + out_pos
            out_pos = pos + out_vel

        return (out_x, (out_pos, out_vel))

    def message(self, x_i: OptTensor, x_j: OptTensor, pos_i: Tensor,
                pos_j: Tensor,
                edge_attr: OptTensor = None) -> Tuple[Tensor, Tensor]:

        msg = torch.sum((pos_i - pos_j).square(), dim=1, keepdim=True)
        msg = msg if x_j is None else torch.cat([x_j, msg], dim=1)
        msg = msg if x_i is None else torch.cat([x_i, msg], dim=1)
        msg = msg if edge_attr is None else torch.cat([msg, edge_attr], dim=1)
        msg = msg if self.local_nn is None else self.local_nn(msg)

        pos_msg = ((pos_i - pos_j) if self.pos_nn is None else
                   (pos_i - pos_j) * self.pos_nn(msg))
        return (msg, pos_msg)

    def aggregate(self, inputs: Tuple[Tensor, Tensor],
                  index: Tensor) -> Tuple[Tensor, Tensor]:
        return (scatter(inputs[0], index, 0, reduce=self.aggr),
                scatter(inputs[1], index, 0, reduce="mean"))

    def update(self, inputs: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        return inputs

    def __repr__(self):
        return ("{}(local_nn={}, pos_nn={},"
                " vel_nn={},"
                " global_nn={})").format(self.__class__.__name__,
                                         self.local_nn, self.pos_nn,
                                         self.vel_nn, self.global_nn)
