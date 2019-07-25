from ..utils import get_laplacian
from ..utils import to_scipy_sparse_matrix

import torch
from torch_sparse import transpose
from scipy.sparse.linalg import eigs, eigsh
from scipy.sparse.linalg import ArpackError, ArpackNoConvergence


class LaplacianLambdaMax:
    """ Computes the highest eigenvalue of graph Laplacian

    The graph Laplacian here is the symmetrically normalized
    graph Laplacian.

    Args:
        lambda_max (float): set lambda_max for all graphs to this value
        fall_back_lambda_max (float): if the numerical computation
            of the eigenvalues fails, then set lambda_max to this value
    """

    def __init__(self, lambda_max=None, fall_back_lambda_max=2.0):
        self.lambda_max = lambda_max
        self.fall_back_lambda_max = fall_back_lambda_max

    def __call__(self, data):
        if self.lambda_max is not None:
            data.lambda_max = self.lambda_max
            return data

        if data.num_edges is None or data.num_edges == 0:
            return data

        edge_index = data.edge_index
        edge_weight = data.edge_weight if hasattr(data, 'edge_weight') \
            else None
        num_nodes = data.num_nodes if hasattr(data, 'num_nodes') else None

        edge_index, edge_weight = get_laplacian(edge_index,
                                                edge_weight=edge_weight,
                                                num_nodes=num_nodes,
                                                normalization='sym')
        laplacian = to_scipy_sparse_matrix(edge_index,
                                           edge_weight,
                                           num_nodes=num_nodes)

        # Decide which eigenvalue algorithm to used base on whether
        # the Laplacian is symmetric or not
        # This works because both `get_laplacian` and `transpose`
        # sort the `edge_index`
        _, edge_weight_t = transpose(edge_index,
                                     edge_weight,
                                     num_nodes,
                                     num_nodes)
        if data.is_undirected() and torch.allclose(edge_weight, edge_weight_t):
            # Laplacian is symmetric, use
            # eigsh is fast, but only works for symmetric / hermitian matrices
            eig_fn = eigsh
        else:
            eig_fn = eigs  # slower

        try:
            lambda_max = eig_fn(laplacian,
                                k=1,
                                which='LM',
                                return_eigenvectors=False)
            lambda_max = lambda_max[0]
        except (ArpackError, ArpackNoConvergence):
            lambda_max = None

        # Make sure lambda_max is always set
        if lambda_max is None:
            lambda_max = self.fall_back_lambda_max

        data.lambda_max = lambda_max.real

        return data

    def __repr__(self):
        return '{}(lambda_max={})'.format(self.__class__.__name__,
                                          self.lambda_max)
