import math

import torch
from numpy import identity, diagonal
from torch.nn import Parameter

from torch_geometric.nn.inits import zeros
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix


def jacobi(a, iterations=10):  # Jacobi method
    """
    Calculates eigen decomposition using jacobi method
     Taken from
     https://github.com/mateuv/MetodosNumericos/blob/master/python/NumericalMethodsInEngineeringWithPython/jacobi.py
     Modified ending condition
    """

    def maxElem(a):  # Find largest off-diag. element a[k,l]
        n = len(a)
        aMax = 0.0
        for i in range(n - 1):
            for j in range(i + 1, n):
                if abs(a[i, j]) >= aMax:
                    aMax = abs(a[i, j])
                    k = i
                    l = j
        return aMax, k, l

    def rotate(a, p, k, l):  # Rotate to make a[k,l] = 0
        n = len(a)
        aDiff = a[l, l] - a[k, k]
        if abs(a[k, l]) < abs(aDiff) * 1.0e-36:
            t = a[k, l] / aDiff
        else:
            phi = aDiff / (2.0 * a[k, l])
            t = 1.0 / (abs(phi) + math.sqrt(phi ** 2 + 1.0))
            if phi < 0.0: t = -t
        c = 1.0 / math.sqrt(t ** 2 + 1.0);
        s = t * c
        tau = s / (1.0 + c)
        temp = a[k, l]
        a[k, l] = 0.0
        a[k, k] = a[k, k] - t * temp
        a[l, l] = a[l, l] + t * temp
        for i in range(k):  # Case of i < k
            temp = a[i, k]
            a[i, k] = temp - s * (a[i, l] + tau * temp)
            a[i, l] = a[i, l] + s * (temp - tau * a[i, l])
        for i in range(k + 1, l):  # Case of k < i < l
            temp = a[k, i]
            a[k, i] = temp - s * (a[i, l] + tau * a[k, i])
            a[i, l] = a[i, l] + s * (temp - tau * a[i, l])
        for i in range(l + 1, n):  # Case of i > l
            temp = a[k, i]
            a[k, i] = temp - s * (a[l, i] + tau * temp)
            a[l, i] = a[l, i] + s * (temp - tau * a[l, i])
        for i in range(n):  # Update transformation matrix
            temp = p[i, k]
            p[i, k] = temp - s * (p[i, l] + tau * p[i, k])
            p[i, l] = p[i, l] + s * (temp - tau * p[i, l])

    n = len(a)
    p = identity(n) * 1.0  # Initialize transformation matrix
    for i in range(iterations):  # Jacobi rotation loop
        aMax, k, l = maxElem(a)
        rotate(a, p, k, l)
    return diagonal(a), p


def atan_threshold(A):
    """
    Computes 2 * arctan(1/A) with A tensor of shape out_features x in_features x N
    if an entry is zero, returns exactly 2
    """

    eps = 1e-5

    A = torch.where(A > eps, 2.0 * torch.atan(1 / A), 2.0 * torch.ones(A.size()))

    return A


def Pmul(P, X):
    """
    3D x 2D Tensor multiplication :
    tensor P (out x d x in)
    tensor X (d x in)
    output : (out x d x in)
    """

    Nout = P.size()[0]
    d = P.size()[1]
    Nin = P.size()[2]

    output = torch.zeros(Nout, d, Nin)

    for b in range(Nout):
        Pslice = P[b, :, :]  # d x in
        aux = Pslice * X  # d x in

        output[b, :, :] = aux

    return output


class CayleyConv(torch.nn.Module):

    def __init__(self, in_channels, out_channels, r, normalization='sym',
                 bias=True, jacobi_iterations=10):
        """
        Cayley Filter Convolutional Layer from 'CayleyNets: Graph Convolutional Neural Networks
        with Complex Rational Spectral Filters' https://arxiv.org/pdf/1705.07664.pdf

        Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        r (int): Cayley polynomial order :math:`r`.
        normalization (str, optional): The normalization scheme for the graph
            Laplacian (default: :obj:`"sym"`):

            1. :obj:`None`: No normalization
            :math:`\mathbf{L} = \mathbf{D} - \mathbf{A}`

            2. :obj:`"sym"`: Symmetric normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
            \mathbf{D}^{-1/2}`

            3. :obj:`"rw"`: Random-walk normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1} \mathbf{A}`
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        jacobi_iterations (int, default: 10): Amount of iterations for the jacobi method
        """
        super(CayleyConv, self).__init__()

        assert r > 0
        assert normalization in [None, 'sym', 'rw'], 'Invalid normalization'

        self.normalization = normalization
        self.in_channels = in_channels
        self.out_channels = out_channels
        #        Reference Implementation https://github.com/amoliu/CayleyNet/blob/master/CayleyNet.ipynb
        #         Jacobi method for approximate eigen decomposition (default to 10 iterations)
        self.jacobi_iterations = jacobi_iterations

        self.r = r
        self.real_weight = Parameter(torch.Tensor(out_channels, in_channels, r, 1))
        self.imag_weight = Parameter(torch.Tensor(out_channels, in_channels, r, 1))
        self.h = Parameter(torch.Tensor(out_channels, in_channels))  # zoom parameter
        self.c = Parameter(torch.Tensor(out_channels, in_channels, 1))  # coefficients

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.real_weight.size(1))
        self.real_weight.data.uniform_(-stdv, stdv)
        self.imag_weight.data.uniform_(-stdv, stdv)
        self.h.data.uniform_(-stdv, stdv)
        self.c.data.uniform_(-stdv, stdv)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_weight=None):
        """
        Source: https://github.com/WhiteNoyse/SiGCN
        """
        edge_index, edge_weight = get_laplacian(edge_index, edge_weight,
                                                self.normalization)
        L = to_scipy_sparse_matrix(edge_index, edge_weight)

        (w, U) = jacobi(L.todense(), self.jacobi_iterations)

        w = torch.tensor(w)
        Ut = torch.tensor(U.transpose(0, 1)).float()
        aux = torch.mm(Ut, x)

        aux_2 = torch.tensordot(self.h, w, dims=0)
        aux_2 = atan_threshold(aux_2)
        aux_2 = torch.tensordot(aux_2, torch.arange(1, self.r + 1, dtype=torch.float),
                                dims=0)

        aux_cos = torch.cos(aux_2).transpose(3, 2)
        aux_sin = torch.sin(aux_2).transpose(3, 2)
        aux_cos = self.real_weight * aux_cos
        aux_sin = self.imag_weight * aux_sin

        aux_cos = aux_cos - aux_sin
        aux_cos = torch.sum(aux_cos, 2).squeeze()
        aux_cos = aux_cos + self.c
        aux_cos = aux_cos.transpose(2, 1)

        out = Pmul(aux_cos, aux)
        out = torch.sum(out, 2).squeeze()
        out = out.transpose(0, 1)
        output = torch.mm(torch.Tensor(U), out)
        output.clamp(min=0)
        if self.bias is not None:
            output += self.bias
        return output

    def __repr__(self):
        return '{}({}, {}, r={}, normalization={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.r, self.normalization)
