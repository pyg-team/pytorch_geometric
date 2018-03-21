import torch
from torch.nn import Module, Parameter

from .utils.repr import repr
from ..functional.agnn import agnn


class AGNN(Module):
    """Graph Attentional Layer from the `"Attention-based Graph Neural Network
    for Semi-Supervised Learning (AGNN)" <https://arxiv.org/abs/1803.03735>`_
    paper."""

    def __init__(self):
        super(AGNN, self).__init__()
        self.beta = Parameter(torch.Tensor(1))
        self.reset_parameters()

    def reset_parameters(self):
        self.beta.data.uniform_(0, 1)

    def forward(self, x, edge_index):
        return agnn(x, edge_index, self.beta)

    def __repr__(self):
        return repr(self)
