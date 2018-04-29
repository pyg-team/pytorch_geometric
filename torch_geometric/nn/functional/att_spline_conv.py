import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch_spline_conv import SplineBasis, SplineWeighting


def att_spline_conv(x,
                    edge_index,
                    pseudo,
                    weight,
                    root_weight,
                    att_weight,
                    kernel_size,
                    is_open_spline,
                    degree=1,
                    negative_slope=0.2,
                    dropout=0.5,
                    bias=None):

    x = x.unsqueeze(-1) if x.dim() == 1 else x
    row, col = edge_index
    pseudo = pseudo.unsqueeze(-1) if pseudo.dim() == 1 else pseudo

    # Weight root node.
    output = torch.mm(x, root_weight)

    # Weight each node.
    basis, weight_index = SplineBasis.apply(degree, pseudo, kernel_size,
                                            is_open_spline)
    x_col = SplineWeighting.apply(x[col], weight, basis, weight_index)

    # Compute attention across edges.
    x_row = output[row]
    alpha = torch.cat([x_row, x_col], dim=-1)
    alpha = torch.matmul(alpha, att_weight.unsqueeze(-1)).squeeze()
    alpha = F.leaky_relu(alpha, negative_slope)

    # Scatter softmax.
    alpha = alpha.exp_()
    alpha_sum = Variable(alpha.data.new(x.size(0))).fill_(0)
    alpha /= alpha_sum.scatter_add_(0, Variable(row), alpha)[row]

    # Apply attention.
    alpha_dropout = F.dropout(alpha, p=dropout, training=True)
    x_col = alpha_dropout.unsqueeze(-1) * x_col

    # Sum up neighborhoods.
    output.scatter_add_(0, Variable(row).unsqueeze(-1).expand_as(x_col), x_col)

    # Add bias (if wished).
    if bias is not None:
        output += bias

    return output, alpha
