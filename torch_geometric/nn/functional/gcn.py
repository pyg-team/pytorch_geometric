from torch.autograd import Variable

from ...sparse import mm, mm_diagonal, eye


def gcn(adj, input, weight, bias=None):
    n, m = adj.size()

    I = eye(n, m)
    adj = adj + I
    deg = sum(adj, dim=1)
    deg = deg.pow(-0.5)

    adj = mm_diagonal(deg, adj)
    adj = mm_diagonal(deg, adj, transpose=True)

    output = mm(Variable(adj), input)
    output = mm(output, weight)

    if bias is not None:
        output += bias

    return output
