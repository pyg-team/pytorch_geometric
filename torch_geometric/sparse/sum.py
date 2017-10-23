import torch
from torch.autograd import Function


class _Sum(Function):
    def __init__(self, dim):
        super(_Sum, self).__init__()
        self.dim = dim

    def forward(self, a):
        self.save_for_backward(a)
        return _sum(a, self.dim)

    def backward(self, grad_output):
        print(grad_output)
        pass


def _sum(a, dim):
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

    return zero.scatter_add_(0, index, values)


def sum(a, dim=None):
    if torch.is_tensor(a):
        return _sum(a, dim)
    else:
        return _Sum(dim)(a)
