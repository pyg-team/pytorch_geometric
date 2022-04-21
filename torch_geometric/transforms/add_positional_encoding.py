from typing import Optional, Union

from scipy.sparse.linalg import eigs, eigsh
import numpy as np

import torch

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix


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
        if self.name == 'laplacian_eigenvector_pe':
            eig_fn = eigs if not self.is_undirected else eigsh
            edge_index, edge_weight = get_laplacian(
                data.edge_index, normalization='sym', dtype=torch.float,
                num_nodes=data.num_nodes)
            L = to_scipy_sparse_matrix(edge_index, edge_weight, data.num_nodes)
            eig_vals, eig_vecs = eig_fn(L, k=self.num_channels + 1, which='SR',
                                        return_eigenvectors=True)
            eig_vecs = eig_vecs[:, eig_vals.argsort()]
            pe = torch.from_numpy(np.real(eig_vecs[:, 1:self.num_channels + 1]))

        elif self.name == 'random_walk_pe':
            raise NotImplementedError

        for store in data.node_stores:
            if self.method == 'attr':
                setattr(store, self.name, pe)
            elif self.method == 'cat':
                x = store.x.view(-1, 1) if store.x.dim() == 1 else store.x
                store.x = torch.cat([x, pe.to(x.device, x.dtype)], dim=-1)
        return data
