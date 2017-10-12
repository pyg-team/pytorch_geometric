import torch


def sum(a, dim=None, keepdim=False):
    if dim is None:
        return torch.sum(a._values())

    if dim is 0:
        a = a.t()

    ones = torch.ones(a.size()[1], 1)
    if torch.cuda.is_available():
        ones = ones.cuda()

    c = torch.mm(a, ones)

    if dim is 0:
        c = c.view(1, -1)

    return c if keepdim else c.view(-1)
