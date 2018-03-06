import torch
from torch.autograd import Function

from ...sparse import SparseTensor


class SparseMM(Function):
    def __init__(self, sparse):
        super(SparseMM, self).__init__()
        self.sparse = sparse

    def forward(self, dense):
        return torch.mm(self.sparse, dense)

    def backward(self, grad_output):
        grad_input = None
        if self.needs_input_grad[0]:
            grad_input = torch.mm(self.sparse.t(), grad_output)
        return grad_input


def graph_conv(x, index, weight, bias=None):
    row, col = index
    n, e = x.size(0), row.size(0)

    # Preprocess.
    zero, one = x.data.new(n).fill_(0), x.data.new(e).fill_(1)
    degree = zero.scatter_add_(0, row, one) + 1
    degree = degree.pow_(-0.5)

    value = degree[row] * degree[col]
    value = torch.cat([value, degree * degree], dim=0)
    loop = torch.arange(0, n, out=index.new()).view(1, -1).repeat(2, 1)
    index = torch.cat([index, loop], dim=1)
    adj = SparseTensor(index, value, torch.Size([n, n]))

    # Start computation.
    x = SparseMM(adj)(torch.mm(x, weight))

    if bias is not None:
        x += bias

    return x
