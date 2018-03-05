from __future__ import division
from torch.nn.init import xavier_uniform, constant
import math


def uniform(size, tensor):
    stdv = 1 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-stdv, stdv)


def normal(tensor):
    if tensor is not None:
        tensor.data.normal_()


def xavier(tensor):
    if tensor is not None:
        xavier_uniform(tensor)


def const(tensor, val):
    if tensor is not None:
        constant(tensor, val)
