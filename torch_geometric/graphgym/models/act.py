import torch.nn as nn

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_act

if cfg is not None:
    register_act('relu', nn.ReLU(inplace=cfg.mem.inplace))
    register_act('selu', nn.SELU(inplace=cfg.mem.inplace))
    register_act('prelu', nn.PReLU())
    register_act('elu', nn.ELU(inplace=cfg.mem.inplace))
    register_act('lrelu_01', nn.LeakyReLU(0.1, inplace=cfg.mem.inplace))
    register_act('lrelu_025', nn.LeakyReLU(0.25, inplace=cfg.mem.inplace))
    register_act('lrelu_05', nn.LeakyReLU(0.5, inplace=cfg.mem.inplace))
