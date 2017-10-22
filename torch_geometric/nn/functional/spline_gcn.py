import torch

# from torch import nn

from ...sparse import sum
from .spline_utils import spline_weights


def spline_gcn(adj,
               features,
               weight,
               kernel_size,
               max_radius,
               degree=1,
               bias=None):

    values = adj._values()
    indices = adj._indices()
    _, cols = indices

    # Convert to [|E| x M_in] feature matrix and calculate [|E| x M_out].
    output = features[cols]
    output = edgewise_spline_gcn(values, output, weight, kernel_size,
                                 max_radius, degree)

    # Convolution via sparse row sum. Converts [|E| x M_out] feature matrix to
    # [n x M_out] feature matrix.
    adj = torch.sparse.FloatTensor(indices, output, adj.size())
    output = sum(adj, dim=1)

    # TODO: root node and weight mean

    # Only  2d case at the moment
    # root_weight = weight[torch.arange(kernel_size[-1])]
    # root_weight.mean(0)

    if bias is not None:
        output += bias

    return output


def edgewise_spline_gcn(values,
                        features,
                        weight,
                        kernel_size,
                        max_radius,
                        degree=1):

    K, M_in, M_out = weight.size()
    dim = len(kernel_size)
    m = degree + 1

    # Preprocessing.
    amount, index = spline_weights(values, kernel_size, max_radius, degree)

    features_out = torch.zeros(features.size(0), M_out)

    for k in range(m**dim):
        b = amount[:, k]  # [|E|]
        c = index[:, k]  # [|E|]

        for i in range(M_in):
            pass
            w = weight[:, i]  # [K x M_out]
            w = w[c]  # [|E| x M_out]
            f = features[:, i]  # [|E|]
            features_out += (f * b * w.t()).t()  # [|E| x M_out]

    return features_out
