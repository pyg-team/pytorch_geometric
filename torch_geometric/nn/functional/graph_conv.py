import torch
from torch_geometric.utils import degree, add_self_loops, matmul

from torch.autograd import Function


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


def graph_conv(x, edge_index, edge_attr, weight, bias=None):
    row, col = edge_index
    num_nodes, e = x.size(0), row.size(0)

    edge_attr = x.new_full((e, ), 1) if edge_attr is None else edge_attr
    deg = degree(row, num_nodes, dtype=x.dtype, device=x.device).pow_(-0.5)

    # Normalize and append adjacency matrix by self loops.
    edge_attr = deg[row] * edge_attr * deg[col]

    # Append self-loops.
    edge_attr = torch.cat([edge_attr, deg * deg], dim=0)
    edge_index = add_self_loops(edge_index, num_nodes)

    # Perform the convolution.
    out = matmul(edge_index, edge_attr, torch.mm(x, weight))

    if bias is not None:
        out += bias

    return out
