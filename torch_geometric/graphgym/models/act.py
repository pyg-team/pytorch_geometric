import torch.nn as nn

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_act


def relu():
    return nn.ReLU(inplace=cfg.mem.inplace)


def selu():
    return nn.SELU(inplace=cfg.mem.inplace)


def prelu():
    return nn.PReLU()


def elu():
    return nn.ELU(inplace=cfg.mem.inplace)


def lrelu_01():
    return nn.LeakyReLU(0.1, inplace=cfg.mem.inplace)


def lrelu_025():
    return nn.LeakyReLU(0.25, inplace=cfg.mem.inplace)


def lrelu_05():
    return nn.LeakyReLU(0.5, inplace=cfg.mem.inplace)


if cfg is not None:
    register_act('relu', relu)
    register_act('selu', selu)
    register_act('prelu', prelu)
    register_act('elu', elu)
    register_act('lrelu_01', lrelu_01)
    register_act('lrelu_025', lrelu_025)
    register_act('lrelu_05', lrelu_05)
