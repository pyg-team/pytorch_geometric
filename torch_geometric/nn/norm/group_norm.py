from torch_geometric.typing import OptTensor

import torch
from torch.nn import Parameter
from torch import Tensor
from torch_scatter import scatter_add
from torch_geometric.utils import degree
from ..inits import zeros, ones

class GroupNorm(torch.nn.Module):
    r"""Applies group normalization over each group of features and individual example
    in a batch of node features as described in the `"Group Normalization"
    <https://arxiv.org/abs/1803.08494>`_ paper
    .. math::
        \mathbf{x}^{\prime}_i = \frac{\mathbf{x} -
        \textrm{E}[\mathbf{x}]}{\sqrt{\textrm{Var}[\mathbf{x}] + \epsilon}}
        \odot \gamma + \beta
    The mean and standard-deviation are calculated across all nodes separately for
    each object and group of features in a mini-batch.
    Args:
        in_channels (int): Size of each input sample.
        groups (int): Number of groups to seperate :args:`in_channels` into.
            (default: :obj:`2`)
        eps (float, optional): A value added to the denominator for numerical
            stability. (default: :obj:`1e-5`)
        affine (bool, optional): If set to :obj:`True`, this module has
            learnable affine parameters :math:`\gamma` and :math:`\beta`.
            (default: :obj:`True`)
    """
    def __init__(self, in_channels, groups = 2, eps=1e-5, affine=True):
        super(GroupNorm, self).__init__()
        
        assert in_channels%groups == 0
        self.in_channels = in_channels
        self.eps = eps
        self.groups = groups
        self.affine = affine

        if affine:
            self.weight = Parameter(torch.Tensor([in_channels]))
            self.bias = Parameter(torch.Tensor([in_channels]))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        ones(self.weight)
        zeros(self.bias)

    def forward(self, x: Tensor, batch: OptTensor = None)->Tensor:
        """"""
        if batch is None:
            batch = torch.zeros(x.size(0), device = x.device, dtype = torch.long)

        batch_size = int(batch.max()) + 1
        group_size = int(x.size(-1)/self.groups)
        group_id = torch.cat([torch.tensor([i]*group_size, device = x.device)
                              for i in torch.arange(self.groups)])
        norm = (degree(batch, batch_size, dtype = x.dtype).clamp_(min=1.0) * 
                float(group_size)).view(-1,1)

        mean = scatter_add(scatter_add(x, batch, dim = 0, dim_size = batch_size),
                           group_id, dim = -1, dim_size = group_size)/norm
        x = x - mean[:,group_id][batch]

        var = scatter_add(scatter_add(x * x, batch, dim = 0, dim_size = batch_size),
                          group_id, dim = -1, dim_size = group_size)/norm
        out = x / (var+self.eps).sqrt()[:,group_id][batch]
        
        if self.weight is not None and self.bias is not None:
            return out*self.weight + self.bias
        return out

    def __repr__(self):
        return '{}(in_channels={}, groups={}, affine={})'.format(self.__class__.__name__,
                                                                 self.in_channels,
                                                                 self.groups,
                                                                 self.affine)
