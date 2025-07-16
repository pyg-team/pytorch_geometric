from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import (
    add_self_loops,
    coalesce,
    get_ppr,
    is_undirected,
    scatter,
    sort_edge_index,
    to_dense_adj,
)


@functional_transform('gdc')
class GDC(BaseTransform):
    r"""Processes the graph via Graph Diffusion Convolution (GDC) from the
    `"Diffusion Improves Graph Learning" <https://arxiv.org/abs/1911.05485>`_
    paper (functional name: :obj:`gdc`).

    .. note::

        The paper offers additional advice on how to choose the
        hyperparameters.
        For an example of using GCN with GDC, see `examples/gcn.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        gcn.py>`_.

    Args:
        self_loop_weight (float, optional): Weight of the added self-loop.
            Set to :obj:`None` to add no self-loops. (default: :obj:`1`)
        normalization_in (str, optional): Normalization of the transition
            matrix on the original (input) graph. Possible values:
            :obj:`"sym"`, :obj:`"col"`, and :obj:`"row"`.
            See :func:`GDC.transition_matrix` for details.
            (default: :obj:`"sym"`)
        normalization_out (str, optional): Normalization of the transition
            matrix on the transformed GDC (output) graph. Possible values:
            :obj:`"sym"`, :obj:`"col"`, :obj:`"row"`, and :obj:`None`.
            See :func:`GDC.transition_matrix` for details.
            (default: :obj:`"col"`)
        diffusion_kwargs (dict, optional): Dictionary containing the parameters
            for diffusion.
            `method` specifies the diffusion method (:obj:`"ppr"`,
            :obj:`"heat"` or :obj:`"coeff"`).
            Each diffusion method requires different additional parameters.
            See :func:`GDC.diffusion_matrix_exact` or
            :func:`GDC.diffusion_matrix_approx` for details.
            (default: :obj:`dict(method='ppr', alpha=0.15)`)
        sparsification_kwargs (dict, optional): Dictionary containing the
            parameters for sparsification.
            `method` specifies the sparsification method (:obj:`"threshold"` or
            :obj:`"topk"`).
            Each sparsification method requires different additional
            parameters.
            See :func:`GDC.sparsify_dense` for details.
            (default: :obj:`dict(method='threshold', avg_degree=64)`)
        exact (bool, optional): Whether to exactly calculate the diffusion
            matrix.
            Note that the exact variants are not scalable.
            They densify the adjacency matrix and calculate either its inverse
            or its matrix exponential.
            However, the approximate variants do not support edge weights and
            currently only personalized PageRank and sparsification by
            threshold are implemented as fast, approximate versions.
            (default: :obj:`True`)

    :rtype: :class:`torch_geometric.data.Data`
    """
    def __init__(
        self,
        self_loop_weight: float = 1.,
        normalization_in: str = 'sym',
        normalization_out: str = 'col',
        diffusion_kwargs: Optional[Dict[str, Any]] = None,
        sparsification_kwargs: Optional[Dict[str, Any]] = None,
        exact: bool = True,
    ) -> None:
        self.self_loop_weight = self_loop_weight
        self.normalization_in = normalization_in
        self.normalization_out = normalization_out
        self.diffusion_kwargs = diffusion_kwargs or dict(
            method='ppr', alpha=0.15)
        self.sparsification_kwargs = sparsification_kwargs or dict(
            method='threshold', avg_degree=64)
        self.exact = exact

        if self_loop_weight:
            assert exact or self_loop_weight == 1

    @torch.no_grad()
    def forward(self, data: Data) -> Data:
        assert data.edge_index is not None
        edge_index = data.edge_index
        N = data.num_nodes
        assert N is not None

        if data.edge_attr is None:
            edge_weight = torch.ones(edge_index.size(1),
                                     device=edge_index.device)
        else:
            edge_weight = data.edge_attr
            assert self.exact
            assert edge_weight.dim() == 1

        if self.self_loop_weight:
            edge_index, edge_weight = add_self_loops(
                edge_index, edge_weight, fill_value=self.self_loop_weight,
                num_nodes=N)

        edge_index, edge_weight = coalesce(edge_index, edge_weight, N)

        if self.exact:
            edge_index, edge_weight = self.transition_matrix(
                edge_index, edge_weight, N, self.normalization_in)
            diff_mat = self.diffusion_matrix_exact(edge_index, edge_weight, N,
                                                   **self.diffusion_kwargs)
            edge_index, edge_weight = self.sparsify_dense(
                diff_mat, **self.sparsification_kwargs)
        else:
            edge_index, edge_weight = self.diffusion_matrix_approx(
                edge_index, edge_weight, N, self.normalization_in,
                **self.diffusion_kwargs)
            edge_index, edge_weight = self.sparsify_sparse(
                edge_index, edge_weight, N, **self.sparsification_kwargs)

        edge_index, edge_weight = coalesce(edge_index, edge_weight, N)
        edge_index, edge_weight = self.transition_matrix(
            edge_index, edge_weight, N, self.normalization_out)

        data.edge_index = edge_index
        data.edge_attr = edge_weight

        return data

    def transition_matrix(
        self,
        edge_index: Tensor,
        edge_weight: Tensor,
        num_nodes: int,
        normalization: str,
    ) -> Tuple[Tensor, Tensor]:
        r"""Calculate the approximate, sparse diffusion on a given sparse
        matrix.

        Args:
            edge_index (LongTensor): The edge indices.
            edge_weight (Tensor): One-dimensional edge weights.
            num_nodes (int): Number of nodes.
            normalization (str): Normalization scheme:

                1. :obj:`"sym"`: Symmetric normalization
                   :math:`\mathbf{T} = \mathbf{D}^{-1/2} \mathbf{A}
                   \mathbf{D}^{-1/2}`.
                2. :obj:`"col"`: Column-wise normalization
                   :math:`\mathbf{T} = \mathbf{A} \mathbf{D}^{-1}`.
                3. :obj:`"row"`: Row-wise normalization
                   :math:`\mathbf{T} = \mathbf{D}^{-1} \mathbf{A}`.
                4. :obj:`None`: No normalization.

        :rtype: (:class:`LongTensor`, :class:`Tensor`)
        """
        if normalization == 'sym':
            row, col = edge_index
            deg = scatter(edge_weight, col, 0, num_nodes, reduce='sum')
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        elif normalization == 'col':
            _, col = edge_index
            deg = scatter(edge_weight, col, 0, num_nodes, reduce='sum')
            deg_inv = 1. / deg
            deg_inv[deg_inv == float('inf')] = 0
            edge_weight = edge_weight * deg_inv[col]
        elif normalization == 'row':
            row, _ = edge_index
            deg = scatter(edge_weight, row, 0, num_nodes, reduce='sum')
            deg_inv = 1. / deg
            deg_inv[deg_inv == float('inf')] = 0
            edge_weight = edge_weight * deg_inv[row]
        elif normalization is None:
            pass
        else:
            raise ValueError(
                f"Transition matrix normalization '{normalization}' unknown")

        return edge_index, edge_weight

    def diffusion_matrix_exact(  # noqa: D417
        self,
        edge_index: Tensor,
        edge_weight: Tensor,
        num_nodes: int,
        method: str,
        **kwargs: Any,
    ) -> Tensor:
        r"""Calculate the (dense) diffusion on a given sparse graph.
        Note that these exact variants are not scalable. They densify the
        adjacency matrix and calculate either its inverse or its matrix
        exponential.

        Args:
            edge_index (LongTensor): The edge indices.
            edge_weight (Tensor): One-dimensional edge weights.
            num_nodes (int): Number of nodes.
            method (str): Diffusion method:

                1. :obj:`"ppr"`: Use personalized PageRank as diffusion.
                   Additionally expects the parameter:

                   - **alpha** (*float*) - Return probability in PPR.
                     Commonly lies in :obj:`[0.05, 0.2]`.

                2. :obj:`"heat"`: Use heat kernel diffusion.
                   Additionally expects the parameter:

                   - **t** (*float*) - Time of diffusion. Commonly lies in
                     :obj:`[2, 10]`.

                3. :obj:`"coeff"`: Freely choose diffusion coefficients.
                   Additionally expects the parameter:

                   - **coeffs** (*List[float]*) - List of coefficients
                     :obj:`theta_k` for each power of the transition matrix
                     (starting at :obj:`0`).

        :rtype: (:class:`Tensor`)
        """
        if method == 'ppr':
            # α (I_n + (α - 1) A)^-1
            edge_weight = (kwargs['alpha'] - 1) * edge_weight
            edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                     fill_value=1,
                                                     num_nodes=num_nodes)
            mat = to_dense_adj(edge_index, edge_attr=edge_weight).squeeze()
            diff_matrix = kwargs['alpha'] * torch.inverse(mat)

        elif method == 'heat':
            # exp(t (A - I_n))
            edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                     fill_value=-1,
                                                     num_nodes=num_nodes)
            edge_weight = kwargs['t'] * edge_weight
            mat = to_dense_adj(edge_index, edge_attr=edge_weight).squeeze()
            undirected = is_undirected(edge_index, edge_weight, num_nodes)
            diff_matrix = self.__expm__(mat, undirected)

        elif method == 'coeff':
            adj_matrix = to_dense_adj(edge_index,
                                      edge_attr=edge_weight).squeeze()
            mat = torch.eye(num_nodes, device=edge_index.device)

            diff_matrix = kwargs['coeffs'][0] * mat
            for coeff in kwargs['coeffs'][1:]:
                mat = mat @ adj_matrix
                diff_matrix += coeff * mat
        else:
            raise ValueError(f"Exact GDC diffusion '{method}' unknown")

        return diff_matrix

    def diffusion_matrix_approx(  # noqa: D417
        self,
        edge_index: Tensor,
        edge_weight: Tensor,
        num_nodes: int,
        normalization: str,
        method: str,
        **kwargs: Any,
    ) -> Tuple[Tensor, Tensor]:
        r"""Calculate the approximate, sparse diffusion on a given sparse
        graph.

        Args:
            edge_index (LongTensor): The edge indices.
            edge_weight (Tensor): One-dimensional edge weights.
            num_nodes (int): Number of nodes.
            normalization (str): Transition matrix normalization scheme
                (:obj:`"sym"`, :obj:`"row"`, or :obj:`"col"`).
                See :func:`GDC.transition_matrix` for details.
            method (str): Diffusion method:

                1. :obj:`"ppr"`: Use personalized PageRank as diffusion.
                   Additionally expects the parameters:

                   - **alpha** (*float*) - Return probability in PPR.
                     Commonly lies in :obj:`[0.05, 0.2]`.

                   - **eps** (*float*) - Threshold for PPR calculation stopping
                     criterion (:obj:`edge_weight >= eps * out_degree`).
                     Recommended default: :obj:`1e-4`.

        :rtype: (:class:`LongTensor`, :class:`Tensor`)
        """
        if method == 'ppr':
            if normalization == 'sym':
                # Calculate original degrees.
                _, col = edge_index
                deg = scatter(edge_weight, col, 0, num_nodes, reduce='sum')

            edge_index, edge_weight = get_ppr(
                edge_index,
                alpha=kwargs['alpha'],
                eps=kwargs['eps'],
                num_nodes=num_nodes,
            )

            if normalization == 'col':
                edge_index, edge_weight = sort_edge_index(
                    edge_index.flip([0]), edge_weight, num_nodes)

            if normalization == 'sym':
                # We can change the normalization from row-normalized to
                # symmetric by multiplying the resulting matrix with D^{1/2}
                # from the left and D^{-1/2} from the right.
                # Since we use the original degrees for this it will be like
                # we had used symmetric normalization from the beginning
                # (except for errors due to approximation).
                row, col = edge_index
                deg_inv = deg.sqrt()
                deg_inv_sqrt = deg.pow(-0.5)
                deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
                edge_weight = deg_inv[row] * edge_weight * deg_inv_sqrt[col]
            elif normalization in ['col', 'row']:
                pass
            else:
                raise ValueError(
                    f"Transition matrix normalization '{normalization}' not "
                    f"implemented for non-exact GDC computation")

        elif method == 'heat':
            raise NotImplementedError(
                'Currently no fast heat kernel is implemented. You are '
                'welcome to create one yourself, e.g., based on '
                '"Kloster and Gleich: Heat kernel based community detection '
                '(KDD 2014)."')
        else:
            raise ValueError(f"Approximate GDC diffusion '{method}' unknown")

        return edge_index, edge_weight

    def sparsify_dense(  # noqa: D417
        self,
        matrix: Tensor,
        method: str,
        **kwargs: Any,
    ) -> Tuple[Tensor, Tensor]:
        r"""Sparsifies the given dense matrix.

        Args:
            matrix (Tensor): Matrix to sparsify.
            method (str): Method of sparsification. Options:

                1. :obj:`"threshold"`: Remove all edges with weights smaller
                   than :obj:`eps`.
                   Additionally expects one of these parameters:

                   - **eps** (*float*) - Threshold to bound edges at.

                   - **avg_degree** (*int*) - If :obj:`eps` is not given,
                     it can optionally be calculated by calculating the
                     :obj:`eps` required to achieve a given :obj:`avg_degree`.

                2. :obj:`"topk"`: Keep edges with top :obj:`k` edge weights per
                   node (column).
                   Additionally expects the following parameters:

                   - **k** (*int*) - Specifies the number of edges to keep.

                   - **dim** (*int*) - The axis along which to take the top
                     :obj:`k`.

        :rtype: (:class:`LongTensor`, :class:`Tensor`)
        """
        assert matrix.shape[0] == matrix.shape[1]
        N = matrix.shape[1]

        if method == 'threshold':
            if 'eps' not in kwargs.keys():
                kwargs['eps'] = self.__calculate_eps__(matrix, N,
                                                       kwargs['avg_degree'])

            edge_index = (matrix >= kwargs['eps']).nonzero(as_tuple=False).t()
            edge_index_flat = edge_index[0] * N + edge_index[1]
            edge_weight = matrix.flatten()[edge_index_flat]

        elif method == 'topk':
            k, dim = min(N, kwargs['k']), kwargs['dim']
            assert dim in [0, 1]
            sort_idx = torch.argsort(matrix, dim=dim, descending=True)
            if dim == 0:
                top_idx = sort_idx[:k]
                edge_weight = torch.gather(matrix, dim=dim,
                                           index=top_idx).flatten()

                row_idx = torch.arange(0, N, device=matrix.device).repeat(k)
                edge_index = torch.stack([top_idx.flatten(), row_idx], dim=0)
            else:
                top_idx = sort_idx[:, :k]
                edge_weight = torch.gather(matrix, dim=dim,
                                           index=top_idx).flatten()

                col_idx = torch.arange(
                    0, N, device=matrix.device).repeat_interleave(k)
                edge_index = torch.stack([col_idx, top_idx.flatten()], dim=0)
        else:
            raise ValueError(f"GDC sparsification '{method}' unknown")

        return edge_index, edge_weight

    def sparsify_sparse(  # noqa: D417
        self,
        edge_index: Tensor,
        edge_weight: Tensor,
        num_nodes: int,
        method: str,
        **kwargs: Any,
    ) -> Tuple[Tensor, Tensor]:
        r"""Sparsifies a given sparse graph further.

        Args:
            edge_index (torch.Tensor): The edge indices.
            edge_weight (torch.Tensor): One-dimensional edge weights.
            num_nodes (int): Number of nodes.
            method (str): Method of sparsification:

                1. :obj:`"threshold"`: Remove all edges with weights smaller
                   than :obj:`eps`.
                   Additionally expects one of these parameters:

                   - **eps** (*float*) - Threshold to bound edges at.

                   - **avg_degree** (*int*) - If :obj:`eps` is not given,
                     it can optionally be calculated by calculating the
                     :obj:`eps` required to achieve a given :obj:`avg_degree`.

        :rtype: (:class:`LongTensor`, :class:`Tensor`)
        """
        if method == 'threshold':
            if 'eps' not in kwargs.keys():
                kwargs['eps'] = self.__calculate_eps__(
                    edge_weight,
                    num_nodes,
                    kwargs['avg_degree'],
                )

            remaining_edge_idx = (edge_weight >= kwargs['eps']).nonzero(
                as_tuple=False).flatten()
            edge_index = edge_index[:, remaining_edge_idx]
            edge_weight = edge_weight[remaining_edge_idx]
        elif method == 'topk':
            raise NotImplementedError(
                'Sparse topk sparsification not implemented')
        else:
            raise ValueError(f"GDC sparsification '{method}' unknown")

        return edge_index, edge_weight

    def __expm__(self, matrix: Tensor, symmetric: bool) -> Tensor:
        r"""Calculates matrix exponential.

        Args:
            matrix (Tensor): Matrix to take exponential of.
            symmetric (bool): Specifies whether the matrix is symmetric.

        :rtype: (:class:`Tensor`)
        """
        from scipy.linalg import expm

        if symmetric:
            e, V = torch.linalg.eigh(matrix, UPLO='U')
            diff_mat = V @ torch.diag(e.exp()) @ V.t()
        else:
            diff_mat = torch.from_numpy(expm(matrix.cpu().numpy()))
            diff_mat = diff_mat.to(matrix.device, matrix.dtype)
        return diff_mat

    def __calculate_eps__(
        self,
        matrix: Tensor,
        num_nodes: int,
        avg_degree: int,
    ) -> float:
        r"""Calculates threshold necessary to achieve a given average degree.

        Args:
            matrix (Tensor): Adjacency matrix or edge weights.
            num_nodes (int): Number of nodes.
            avg_degree (int): Target average degree.

        :rtype: (:class:`float`)
        """
        sorted_edges = torch.sort(matrix.flatten(), descending=True).values
        if avg_degree * num_nodes > len(sorted_edges):
            return -np.inf

        left = sorted_edges[avg_degree * num_nodes - 1]
        right = sorted_edges[avg_degree * num_nodes]
        return float(left + right) / 2.0
