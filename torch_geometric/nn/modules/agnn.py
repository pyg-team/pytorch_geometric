import torch
from torch.nn import Module, Parameter

from .utils.repr import repr
from ..functional.agnn import agnn


class AGNN(Module):
    """Graph Attentional Layer from the `"Attention-based Graph Neural Network
    for Semi-Supervised Learning (AGNN)" <https://arxiv.org/abs/1803.03735>`_
    paper.

    Args:
        requires_grad (bool, optional): If set to :obj:`False`, the propagation
            layer will not be trainable. (default: :obj:`True`)
    """

    def __init__(self, requires_grad=True):
        super(AGNN, self).__init__()

        if requires_grad:
            self.beta = Parameter(torch.Tensor(1))
        else:
            self.register_buffer('beta', torch.ones(1))

        self.requires_grad = requires_grad
        self.reset_parameters()

    def reset_parameters(self):
        if self.requires_grad:
            self.beta.data.uniform_(0, 1)

    def forward(self, x, edge_index):
        beta = self.beta if self.requires_grad else self._buffers['beta']
        return agnn(x, edge_index, beta)

    def __repr__(self):
        return repr(self)
