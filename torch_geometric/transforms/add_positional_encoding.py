from typing import Any, Optional

import numpy as np
import torch
from scipy.sparse.linalg import eigs, eigsh
from torch_scatter import scatter_add
from torch_sparse import spspmm

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import (
    full_self_loop_attr,
    get_laplacian,
    to_scipy_sparse_matrix,
)


def add_node_attr(data: Data, value: Any, cat: bool,
                  attr_name: Optional[str] = None):
    # todo: moving this to BaseTransform.
    for store in data.node_stores:
        if cat:
            if hasattr(store, 'x'):
                x = store.x.view(-1, 1) if store.x.dim() == 1 else store.x
                store.x = torch.cat([x, value.to(x.device, x.dtype)], dim=-1)
            else:
                setattr(store, 'x', value)
        else:
            assert attr_name is not None
            setattr(store, attr_name, value)
    return data


@functional_transform('add_laplacian_eigenvector_pe')
class AddLaplacianEigenvectorPE(BaseTransform):
    r"""Adds Laplacian eigenvector positional encoding from the `"Benchmarking
    Graph Neural Networks" <https://arxiv.org/abs/2003.00982>`_ paper to the
    given graph (functional name: :obj:`add_laplacian_eigenvector_pe`).

    Args:
        num_channels (int): The number of channels. It will be the number of
            the smallest non-trivial eigenvectors.
        method (str, optional): The method to add positional encoding.
            If set to :obj:`"attr"`, will be added to the data object
            as a new attribute.
            If set to :obj:`"cat"`, will be concatenated to the existing
            :obj:`x` in the data object. If there is no :obj:`x`, will be
            added as a new :obj:`x`. (default: :obj:`"attr"`)
        attr_name (str, optional): The attribute name of the positional
            encoding to be added when :attr:`"method"` is set to :obj:`"attr"`.
            (default: :obj:`"laplacian_eigenvector_pe"`)
        is_undirected (bool, optional): If set to :obj:`True`, this transform
            expects undirected graphs as input, and can hence speed up the
            computation of the eigenvectors. (default: :obj:`False`)
        **kwargs (optional): The keyword arguments that are passed down to the
            :meth:`scipy.sparse.linalg.eigs` (when :attr:`"is_undirected"` is
            False) or :meth:`scipy.sparse.linalg.eigsh` (when
            :attr:`"is_undirected"` is True) methods for computing
            eigenvectors.
    """
    def __init__(self, num_channels: int, method: Optional[str] = 'attr',
                 attr_name: Optional[str] = 'laplacian_eigenvector_pe',
                 is_undirected: Optional[bool] = False, **kwargs):
        self.num_channels = num_channels
        self.method = method
        self.attr_name = attr_name
        self.is_undirected = is_undirected
        self.kwargs = kwargs

    def __call__(self, data: Data) -> Data:
        N = data.num_nodes
        eig_fn = eigs if not self.is_undirected else eigsh
        eig_which = 'SR' if not self.is_undirected else 'SA'
        edge_index, edge_weight = get_laplacian(data.edge_index,
                                                normalization='sym',
                                                dtype=torch.float, num_nodes=N)
        L = to_scipy_sparse_matrix(edge_index, edge_weight, N)
        eig_vals, eig_vecs = eig_fn(L, k=self.num_channels + 1,
                                    which=eig_which,
                                    return_eigenvectors=True,
                                    **self.kwargs)
        eig_vecs = np.real(eig_vecs[:, eig_vals.argsort()])
        pe = torch.from_numpy(eig_vecs[:, 1:self.num_channels + 1])

        sign = -1. + 2 * torch.randint(0, 2, (self.num_channels, ),
                                       dtype=torch.float)
        pe *= sign

        data = add_node_attr(data, pe, cat=self.method == 'cat',
                             attr_name=self.attr_name)
        return data


@functional_transform('add_random_walk_pe')
class AddRandomWalkPE(BaseTransform):
    r"""Adds random walk positional encoding from the `"Graph Neural Networks
    with Learnable Structural and Positional Representations"
    <https://arxiv.org/abs/2110.07875>`_ paper to the given graph
    (functional name: :obj:`add_random_walk_pe`).

    Args:
        num_channels (int): The number of channels. It will be the number of
            the random walk steps.
        method (str, optional): The method to add positional encoding.
            If set to :obj:`"attr"`, will be added to the data object
            as a new attribute.
            If set to :obj:`"cat"`, will be concatenated to the existing
            :obj:`x` in the data object. If there is no :obj:`x`, will be
            added as a new :obj:`x`. (default: :obj:`"attr"`)
        attr_name (str, optional): The attribute name of the positional
            encoding to be added when :attr:`"method"` is set to :obj:`"attr"`.
            (default: :obj:`"random_walk_pe"`)
    """
    def __init__(self, num_channels: int, method: Optional[str] = 'attr',
                 attr_name: Optional[str] = 'random_walk_pe'):
        self.num_channels = num_channels
        self.method = method
        self.attr_name = attr_name

    def __call__(self, data: Data) -> Data:
        # Compute D^{-1} A.
        N = data.num_nodes
        edge_index, edge_weight = data.edge_index, data.edge_weight
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), dtype=torch.float,
                                     device=edge_index.device)
        row = edge_index[0]
        deg_inv = 1.0 / scatter_add(edge_weight, row, dim=0, dim_size=N)
        deg_inv.masked_fill_(deg_inv == float('inf'), 0)
        edge_weight = deg_inv[row] * edge_weight

        rw_index, rw_weight = edge_index, edge_weight
        pe_list = [full_self_loop_attr(rw_index, rw_weight, num_nodes=N)]
        for _ in range(self.num_channels - 1):
            rw_index, rw_weight = spspmm(rw_index, rw_weight, edge_index,
                                         edge_weight, N, N, N, coalesced=True)
            pe_list.append(
                full_self_loop_attr(rw_index, rw_weight, num_nodes=N))
        pe = torch.stack(pe_list, dim=-1)

        data = add_node_attr(data, pe, cat=self.method == 'cat',
                             attr_name=self.attr_name)
        return data
