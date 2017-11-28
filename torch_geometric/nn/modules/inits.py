import math


def uniform(size, weight, bias=None):
    stdv = 1. / math.sqrt(size)

    weight.data.uniform_(-stdv, stdv)
    if bias is not None:
        bias.data.uniform_(-stdv, stdv)
