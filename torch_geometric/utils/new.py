import torch
from torch.autograd import Variable


def new(x, *size):
    return x.new(*size) if torch.is_tensor(x) else Variable(x.data.new(*size))
