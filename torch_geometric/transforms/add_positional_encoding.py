from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor

import torch_geometric.typing
from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import (
    get_laplacian,
    get_self_loop_attr,
    is_torch_sparse_tensor,
    scatter,
    to_edge_index,
    to_scipy_sparse_matrix,
    to_torch_coo_tensor,
    to_torch_csr_tensor,
)


def add_node_attr(
    data: Data,
    value: Any,
    attr_name: Optional[str] = None,
) -> Data:
    # TODO Move to `BaseTransform`.
    if attr_name is None:
        if data.x is not None:
            x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat([x, value.to(x.device, x.dtype)], dim=-1)
        else:
            data.x = value
    else:
        data[attr_name] = value

    return data


@functional_transform('add_laplacian_eigenvector_pe')
class AddLaplacianEigenvectorPE(BaseTransform):
    r"""Adds the Laplacian eigenvector positional encoding from the
    `"Benchmarking Graph Neural Networks" <https://arxiv.org/abs/2003.00982>`_
    paper to the given graph
    (functional name: :obj:`add_laplacian_eigenvector_pe`).

    Args:
        k (int): The number of non-trivial eigenvectors to consider.
        attr_name (str, optional): The attribute name of the data object to add
            positional encodings to. If set to :obj:`None`, will be
            concatenated to :obj:`data.x`.
            (default: :obj:`"laplacian_eigenvector_pe"`)
        is_undirected (bool, optional): If set to :obj:`True`, this transform
            expects undirected graphs as input, and can hence speed up the
            computation of eigenvectors. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :meth:`scipy.sparse.linalg.eigs` (when :attr:`is_undirected` is
            :obj:`False`) or :meth:`scipy.sparse.linalg.eigsh` (when
            :attr:`is_undirected` is :obj:`True`).
    """
    # Number of nodes from which to use sparse eigenvector computation:
    SPARSE_THRESHOLD: int = 100

    def __init__(
        self,
        k: int,
        attr_name: Optional[str] = 'laplacian_eigenvector_pe',
        is_undirected: bool = False,
        **kwargs: Any,
    ) -> None:
        self.k = k
        self.attr_name = attr_name
        self.is_undirected = is_undirected
        self.kwargs = kwargs

    def forward(self, data: Data) -> Data:
        assert data.edge_index is not None
        num_nodes = data.num_nodes
        assert num_nodes is not None

        edge_index, edge_weight = get_laplacian(
            data.edge_index,
            data.edge_weight,
            normalization='sym',
            num_nodes=num_nodes,
        )

        L = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes)

        if num_nodes < self.SPARSE_THRESHOLD:
            from numpy.linalg import eig, eigh
            eig_fn = eig if not self.is_undirected else eigh

            eig_vals, eig_vecs = eig_fn(L.todense())
        else:
            from scipy.sparse.linalg import eigs, eigsh
            eig_fn = eigs if not self.is_undirected else eigsh

            eig_vals, eig_vecs = eig_fn(
                L,
                k=self.k + 1,
                which='SR' if not self.is_undirected else 'SA',
                return_eigenvectors=True,
                **self.kwargs,
            )

        eig_vecs = np.real(eig_vecs[:, eig_vals.argsort()])
        pe = torch.from_numpy(eig_vecs[:, 1:self.k + 1])
        sign = -1 + 2 * torch.randint(0, 2, (self.k, ))
        pe *= sign

        data = add_node_attr(data, pe, attr_name=self.attr_name)
        return data


@functional_transform('add_random_walk_pe')
class AddRandomWalkPE(BaseTransform):
    r"""Adds the random walk positional encoding from the `"Graph Neural
    Networks with Learnable Structural and Positional Representations"
    <https://arxiv.org/abs/2110.07875>`_ paper to the given graph
    (functional name: :obj:`add_random_walk_pe`).

    Args:
        walk_length (int): The number of random walk steps.
        attr_name (str, optional): The attribute name of the data object to add
            positional encodings to. If set to :obj:`None`, will be
            concatenated to :obj:`data.x`.
            (default: :obj:`"random_walk_pe"`)
    """
    def __init__(
        self,
        walk_length: int,
        attr_name: Optional[str] = 'random_walk_pe',
    ) -> None:
        self.walk_length = walk_length
        self.attr_name = attr_name

    def forward(self, data: Data) -> Data:
        assert data.edge_index is not None
        row, col = data.edge_index
        N = data.num_nodes
        assert N is not None

        if data.edge_weight is None:
            value = torch.ones(data.num_edges, device=row.device)
        else:
            value = data.edge_weight
        value = scatter(value, row, dim_size=N, reduce='sum').clamp(min=1)[row]
        value = 1.0 / value

        if N <= 2_000:  # Dense code path for faster computation:
            adj = torch.zeros((N, N), device=row.device)
            adj[row, col] = value
            loop_index = torch.arange(N, device=row.device)
        elif torch_geometric.typing.NO_MKL:  # pragma: no cover
            adj = to_torch_coo_tensor(data.edge_index, value, size=data.size())
        else:
            adj = to_torch_csr_tensor(data.edge_index, value, size=data.size())

        def get_pe(out: Tensor) -> Tensor:
            if is_torch_sparse_tensor(out):
                return get_self_loop_attr(*to_edge_index(out), num_nodes=N)
            return out[loop_index, loop_index]

        out = adj
        pe_list = [get_pe(out)]
        for _ in range(self.walk_length - 1):
            out = out @ adj
            pe_list.append(get_pe(out))

        pe = torch.stack(pe_list, dim=-1)
        data = add_node_attr(data, pe, attr_name=self.attr_name)

        return data
