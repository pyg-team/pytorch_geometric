import torch
from torch.autograd import Variable


def sum(a, dim=None):
    if dim is None:
        return torch.sum(a._values())

    if dim > 1:
        raise ValueError()

    values = a._values()
    index = a._indices()[1 - dim]

    # TODO: zero as same type as `a`.
    if len(a.size()) == 2:
        zero = torch.zeros(a.size(dim))
    else:
        remaining_dim = list(a.size()[2:])
        zero = torch.zeros(a.size(dim), *remaining_dim)
        index = index.view(-1, 1).expand(index.size(0), *remaining_dim)

    if a.is_cuda is True:
        zero = zero.cuda()

    if not torch.is_tensor(a):
        zero = Variable(zero)

    return zero.scatter_add_(0, index, values)
