import torch

# from torch import nn

from ...sparse import sum
from .spline_utils import spline_weights


def spline_gcn(adj, features, weight, kernel_size, degree=1, bias=None):
    values = adj._values()
    indices = adj._indices()
    _, cols = indices

    # Convert to [|E| x M_in] feature matrix and calculate [|E| x M_out].
    output = features[cols]
    output = edgewise_spline_gcn(values, output, weight, kernel_size, degree)

    # Convolution via sparse row sum. Converts [|E| x M_out] feature matrix to
    # [n x M_out] feature matrix.
    adj = torch.sparse.FloatTensor(indices, output, adj.size())
    output = sum(adj, dim=1)

    if bias is not None:
        output += bias

    return output


def edgewise_spline_gcn(values, features, weight, kernel_size, degree=1):
    K, M_in, M_out = weight.size()

    # Preprocessing.
    amount, index = spline_weights(values, kernel_size, degree)  # [|E| x K]

    features_out = torch.zeros(features.size(0), M_out)
    for k in range(K):
        c = index[:, k]  # [|E|]
        b = amount[:, k]  # [|E|]
        for i in range(M_in):
            w = weight[:, i]  # [K x M_out]
            w = w[c]  # [|E| x M_out]
            f = features[:, i]  # [|E|]
            features_out += f * b * w  # [|E| x M_out]

    return features_out
