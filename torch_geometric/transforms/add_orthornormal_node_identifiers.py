import torch
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.nn.attention.performer import orthogonal_matrix
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import get_laplacian, to_dense_adj


@functional_transform("add_orthonormal_node_identifiers")
class AddOrthonormalNodeIdentifiers(BaseTransform):
    r"""Adds orthonormal node identifiers to a given input graph as described
    in the `"Pure Transformers are Powerful Graph Learners"
    <https://arxiv.org/pdf/2207.02505>` paper.
    (functional name: :obj:`add_orthonormal_node_identifiers`).

    Args:
        d_p (int): Dimension of node identifiers. If using Laplacian
            identifiers and d_p is smaller than the number of nodes in the
            graph, the eigenvectors corresponding to the d_p smallest
            eigenvalues are used. If using orthogonal random features (ORFs)
            and d_p is smaller than the number of nodes in the graph, we
            randomly pick d_p channels. If d_p is larger than the number of
            nodes in the graph, we zero pad channels.
        use_laplacian (bool): If :obj:`True`, use eigenvectors of Laplacian
            matrix as node identifiers. Otherwise, use ORFs.
    """
    def __init__(self, d_p: int, use_laplacian: bool = True):
        self._d_p = d_p
        self._use_laplacian = use_laplacian

    def forward(self, data: Data) -> Data:
        n = data.num_nodes
        if self._use_laplacian is True:
            node_ids = self._get_lap_eigenvectors(data.edge_index, n)
        else:
            node_ids = orthogonal_matrix(n, n)

        if n < self._d_p:
            node_ids = F.pad(node_ids, (0, self._d_p - n), value=0.0)
        else:
            node_ids = node_ids[:, :self._d_p]
        node_ids = F.normalize(node_ids, p=2, dim=1)

        data["node_ids"] = node_ids
        return data

    @staticmethod
    def _get_lap_eigenvectors(edge_index: Tensor, n: int) -> Tensor:
        lap_edge_index, lap_edge_attr = get_laplacian(
            edge_index,
            normalization="sym",
            num_nodes=n,
        )
        lap_mat = to_dense_adj(lap_edge_index, edge_attr=lap_edge_attr)[0]
        _, eigenvectors = torch.linalg.eigh(lap_mat)
        return eigenvectors
