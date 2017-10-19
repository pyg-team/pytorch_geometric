import torch

from ...sparse import mm, sum


def gcn(adj, features, weight, bias):
    # TODO: add identy
    # TODO: Compute degree and normalized adj
    # TODO: Check if on cuda?

    # Calculate D^{-1/2} in vector form.
    # degree = sum(adj, dim=1) + 1
    # degree = degree.pow(-0.5)  # TODO: test if it works with zeros.

    # adj = sparse_tensor_diag_matmul(adj, degree, transpose=True)
    # adj = sparse_tensor_diag_matmul(adj, degree, transpose=False)

    output = mm(adj, features)

    # features = tf.transpose(features)
    # features = tf.multiply(tf.multiply(degree, features), degree)
    # features = tf.transpose(features)
    output += features

    output = torch.mm(output, weight)

    if bias is not None:
        output += bias

    return output
