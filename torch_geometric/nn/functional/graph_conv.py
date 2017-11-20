from torch.autograd import Variable

from ...sparse import mm, mm_diagonal, sum, eye


def graph_conv(adj, input, weight, bias=None):
    n, m = adj.size()

    i = eye(n, m)
    i = i.cuda() if adj.is_cuda else i
    adj = adj + i

    deg = sum(adj, dim=1)
    deg = deg.pow(-0.5)

    adj = mm_diagonal(deg, adj)
    adj = mm_diagonal(deg, adj, transpose=True)

    output = mm(Variable(adj), input)
    output = mm(output, weight)

    if bias is not None:
        output += bias

    return output
