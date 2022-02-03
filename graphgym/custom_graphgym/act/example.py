import torch
import torch.nn as nn

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_act


class SWISH(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x.mul_(torch.sigmoid(x))
            return x
        else:
            return x * torch.sigmoid(x)


register_act('swish', SWISH(inplace=cfg.mem.inplace))
register_act('lrelu_03', nn.LeakyReLU(0.3, inplace=cfg.mem.inplace))
