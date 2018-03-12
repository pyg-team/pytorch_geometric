import torch
from torch.autograd import Variable
import torch.nn.functional as F


def mlp_conv(
        adj,  # Pytorch Tensor (!bp_to_adj) or Pytorch Variable (bp_to_adj)
        input,  # Pytorch Variable
        linears,  # FC Layers for kernel
        root_weight,
        out_features,
        bias=None):

    if input.dim() == 1:
        input = input.unsqueeze(1)

    values = adj['values']
    row, col = adj['indices']

    # Get features for every end vertex with shape [|E| x M_in].
    output = input[col]

    # Sample weights from continuous kernel
    x = values
    for linear in linears[:-1]:
        x = F.relu(linear(x))
    x = linears[-1](x)
    x = x.view(-1, out_features, input.size(1))


    # Compute new Edge-wise Features
    output = torch.matmul(x, output.unsqueeze(2)).squeeze(2)


    # Convolution via `scatter_add`. Converts [|E| x M_out] feature matrix to
    # [n x M_out] feature matrix.
    zero = output.data.new(adj['size'][1], output.size(1)).fill_(0.0)
    zero = Variable(zero) if not torch.is_tensor(output) else zero
    r = row.view(-1, 1).expand(row.size(0), output.size(1))
    output = zero.scatter_add_(0, Variable(r), output)

    # Weighten root node features by multiplying with root weight.
    output += torch.mm(input, root_weight)

    # Normalize output by degree.
    ones = output.data.new(values.size(0)).fill_(1)
    zero = output.data.new(output.size(0)).fill_(0)
    degree = zero.scatter_add_(0, row, ones)
    degree = torch.clamp(degree, min=1)
    output = output / Variable(degree.view(-1, 1))

    if bias is not None:
        output += bias

    return output
