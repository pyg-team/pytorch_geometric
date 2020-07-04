from typing import Optional
from torch_geometric.typing import Adj, OptTensor

import torch
from torch import Tensor
from torch.nn import Parameter
import torch.nn.functional as F
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax


class AGNNConv(MessagePassing):
    r"""The graph attentional propagation layer from the
    `"Attention-based Graph Neural Network for Semi-Supervised Learning"
    <https://arxiv.org/abs/1803.03735>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \mathbf{P} \mathbf{X},

    where the propagation matrix :math:`\mathbf{P}` is computed as

    .. math::
        P_{i,j} = \frac{\exp( \beta \cdot \cos(\mathbf{x}_i, \mathbf{x}_j))}
        {\sum_{k \in \mathcal{N}(i)\cup \{ i \}} \exp( \beta \cdot
        \cos(\mathbf{x}_i, \mathbf{x}_k))}

    with trainable parameter :math:`\beta`.

    Args:
        requires_grad (bool, optional): If set to :obj:`False`, :math:`\beta`
            will not be trainable. (default: :obj:`True`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, requires_grad: bool = True, add_self_loops: bool = True,
                 **kwargs):
        super(AGNNConv, self).__init__(aggr='add', **kwargs)

        self.requires_grad = requires_grad
        self.add_self_loops = add_self_loops

        if requires_grad:
            self.beta = Parameter(torch.Tensor(1))
        else:
            self.register_buffer('beta', torch.ones(1))

        self.reset_parameters()

    def reset_parameters(self):
        if self.requires_grad:
            self.beta.data.fill_(1)

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        """"""
        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index,
                                               num_nodes=x.size(self.node_dim))
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        x_norm = F.normalize(x, p=2., dim=-1)

        # propagate_type: (x: Tensor, x_norm: Tensor)
        return self.propagate(edge_index, x=x, x_norm=x_norm, size=None)

    def message(self, x_j: Tensor, x_norm_i: Tensor, x_norm_j: Tensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        alpha = self.beta * (x_norm_i * x_norm_j).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, size_i)
        return x_j * alpha.view(-1, 1)

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
