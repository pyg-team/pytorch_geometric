import torch
from torch.autograd import Variable

from torch_geometric.sparse import mm


def sum(a, dim=None, keepdim=False):
    if dim is None:
        return torch.sum(a._values())

    if dim is 0:
        a = a.t()

    ones = torch.ones(a.size()[1], 1)
    if a.is_cuda is True:
        ones = ones.cuda()

    if not torch.is_tensor(a):
        ones = Variable(ones)

    c = mm(a, ones)

    if dim is 0:
        c = c.view(1, -1)

    return c if keepdim else c.view(-1)
