from typing import Optional, Union

from scipy.sparse.linalg import eigs, eigsh
import numpy as np

import torch
from torch_scatter import scatter_add
from torch_sparse import spspmm

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix, remove_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes


def diagonal_weight(edge_index, edge_weight, num_nodes=None):
    loop_mask = edge_index[0] == edge_index[1]
    loop_index = edge_index[:, loop_mask][0]
    loop_edge_attr = edge_weight[loop_mask]

    N = maybe_num_nodes(edge_index, num_nodes)
    diag_attr = torch.zeros((N,), dtype=edge_weight.dtype)
    diag_attr[loop_index] = loop_edge_attr
    return diag_attr


@functional_transform('add_positional_encoding')
class AddPositionalEncoding(BaseTransform):
    r"""Adds positional encoding to the given graph (functional name:
    :obj:`add_positional_encoding`).

    Args:
        name: (str): The type of the positional encoding
            (:obj:`"laplacian_eigenvector_pe"`, :obj:`"random_walk_pe"`).
            If set to :obj:`"laplacian_eigenvector_pe"`, the Laplacian
            eigenvector positional encoding from the `"Benchmarking
            Graph Neural Networks" <https://arxiv.org/abs/2003.00982>`_
            paper.
            If set to :obj:`"random_walk_pe"`, the random walk positional
            encoding from the `"Graph Neural Networks with Learnable
            Structural and Positional Representations"
            <https://arxiv.org/abs/2110.07875>`_ paper.
        num_channels: (int): The number of channels. If :attr:`name` is
            :obj:`"laplacian_eigenvector_pe"`, it will be the number of the
            smallest non-trivial eigenvectors. If :attr:`name` is
            :obj:`"random_walk_pe"`, it will be the number of the random walk
            steps.
        method: (str, optional): The method to add positional encoding.
            If set to :obj:`"attr"`, a new attribute will be added to the
                data object.
            If set to :obj:`"cat"`, will be concatenated to the existing
                :obj:`x` in the data object. (default: :obj:`attr`)
        is_undirected (bool, optional): If set to :obj:`True`, this transform
            expects undirected graphs as input, and can hence speed up the
            computation of the eigenvectors. (default: :obj:`False`)
    """

    def __init__(self, name: str, num_channels: int, method: Optional[str] = 'attr',
                 is_undirected: Optional[bool] = False):
        self.name = name
        self.num_channels = num_channels
        self.method = method
        self.is_undirected = is_undirected
        assert name in ['laplacian_eigenvector_pe', 'random_walk_pe']
        assert method in ['attr', 'cat']

    def __call__(self, data: Data):
        pe = None
        N = data.num_nodes
        if self.name == 'laplacian_eigenvector_pe':
            eig_fn = eigs if not self.is_undirected else eigsh
            edge_index, edge_weight = get_laplacian(
                data.edge_index, normalization='sym', dtype=torch.float,
                num_nodes=N)
            L = to_scipy_sparse_matrix(edge_index, edge_weight, N)
            eig_vals, eig_vecs = eig_fn(L, k=self.num_channels + 1, which='SR',
                                        return_eigenvectors=True)
            eig_vecs = eig_vecs[:, eig_vals.argsort()]
            pe = torch.from_numpy(np.real(eig_vecs[:, 1:self.num_channels + 1]))

            sign = 2 * torch.randint(0, 2, (self.num_channels, ), dtype=torch.float) - 1
            pe *= sign

        elif self.name == 'random_walk_pe':
            # Compute D^{-1} A.
            edge_index, edge_weight = remove_self_loops(data.edge_index, data.edge_weight)
            if edge_weight is None:
                edge_weight = torch.ones(edge_index.size(1), dtype=torch.float,
                                         device=edge_index.device)
            row, col = edge_index[0], edge_index[1]
            deg_inv = 1.0 / scatter_add(edge_weight, row, dim=0, dim_size=N)
            deg_inv.masked_fill_(deg_inv == float('inf'), 0)
            edge_weight = deg_inv[row] * edge_weight

            rw_index, rw_weight = edge_index, edge_weight
            pe_list = [diagonal_weight(rw_index, rw_weight)]
            for _ in range(self.num_channels - 1):
                rw_index, rw_weight = spspmm(rw_index, rw_weight, edge_index, edge_weight,
                                             N, N, N, coalesced=True)
                pe_list.append(diagonal_weight(rw_index, rw_weight))
            pe = torch.stack(pe_list, dim=-1)

        for store in data.node_stores:
            if self.method == 'attr':
                setattr(store, self.name, pe)
            elif self.method == 'cat':
                x = store.x.view(-1, 1) if store.x.dim() == 1 else store.x
                store.x = torch.cat([x, pe.to(x.device, x.dtype)], dim=-1)
        return data
