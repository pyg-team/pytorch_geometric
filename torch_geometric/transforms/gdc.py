import torch
import numba
import numpy as np
from scipy.linalg import expm
from torch_geometric.utils import add_self_loops, is_undirected, to_dense_adj
from torch_sparse import coalesce, spspmm
from torch_scatter import scatter_add


def jit():
    def decorator(func):
        try:
            return numba.jit(cache=True)(func)
        except RuntimeError:
            return numba.jit(cache=False)(func)

    return decorator


class GDC(object):
    r"""Processes the graph via Graph Diffusion Convolution (GDC) from the
    `"Diffusion Improves Graph Learning" <https://www.kdd.in.tum.de/gdc>`_
    paper.

    .. note::

        The paper offers additional advice on how to choose the
        hyperparameters.
        For an example of using GCN with GDC, see `:source:`examples/gcn.py`
        <https://github.com/rusty1s/pytorch_geometric/blob/master/examples/
        gcn.py>`_.

    Args:
        self_loop_weight (float, optional): Weight of the added self-loop.
            Set to :obj:`None` to add no self-loops. (default: :obj:`1`)
        normalization_in (str, optional): Normalization of the transition
            matrix on the original (input) graph. Possible values:
            `"sym"`, `"col"`, and `"row"`.
            See :func:`GDC.transition_matrix` for details.
            (default: :obj:`"sym"`)
        normalization_out (str, optional): Normalization of the transition
            matrix on the transformed GDC (output) graph. Possible values:
            `"sym"`, `"col"`, `"row"`, and `None`.
            See :func:`GDC.transition_matrix` for details.
            (default: :obj:`"col"`)
        diffusion_kwargs (dict, optional): Dictionary containing the parameters
            for diffusion.
            `method` specifies the diffusion method (`"ppr"`, `"heat"` or
            `"coeff"`).
            Each diffusion method requires different additional parameters.
            See :func:`GDC.diffusion_matrix_exact` or
            :func:`GDC.diffusion_matrix_approx` for details.
            (default: :obj:`dict(method='ppr', alpha=0.15)`)
        sparsification_kwargs (dict, optional): Dictionary containing the
            parameters for sparsification.
            `method` specifies the sparsification method (`threshold` or
            `topk`).
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
    def __init__(self, self_loop_weight=1, normalization_in='sym',
                 normalization_out='col',
                 diffusion_kwargs=dict(method='ppr', alpha=0.15),
                 sparsification_kwargs=dict(method='threshold',
                                            avg_degree=64), exact=True):
        self.self_loop_weight = self_loop_weight
        self.normalization_in = normalization_in
        self.normalization_out = normalization_out
        self.diffusion_kwargs = diffusion_kwargs
        self.sparsification_kwargs = sparsification_kwargs
        self.exact = exact

        if self_loop_weight:
            assert exact or self_loop_weight == 1

    @torch.no_grad()
    def __call__(self, data):
        N = data.num_nodes
        edge_index = data.edge_index
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

        edge_index, edge_weight = coalesce(edge_index, edge_weight, N, N)

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

        edge_index, edge_weight = coalesce(edge_index, edge_weight, N, N)
        edge_index, edge_weight = self.transition_matrix(
            edge_index, edge_weight, N, self.normalization_out)

        data.edge_index = edge_index
        data.edge_attr = edge_weight

        return data

    def transition_matrix(self, edge_index, edge_weight, num_nodes,
                          normalization):
        r"""Calculate the approximate, sparse diffusion on a given sparse
        matrix.

        Args:
            edge_index (LongTensor): The edge indices.
            edge_weight (Tensor): One-dimensional edge weights.
            num_nodes (int): Number of nodes.
            normalization (str): Normalization scheme. Options:
                1. `"sym"`: Symmetric normalization:
                :math:`\mathbf{T} = \mathbf{D}^{-1/2} \mathbf{A}
                \mathbf{D}^{-1/2}`

                2. `"col"`: Column-wise normalization:
                :math:`\mathbf{T} = \mathbf{A} \mathbf{D}^{-1}`

                3. `"row"`: Row-wise normalization:
                :math:`\mathbf{T} = \mathbf{D}^{-1} \mathbf{A}`

                4. `None`: No normalization.

        :rtype: (:class:`LongTensor`, :class:`Tensor`)
        """
        diag_idx = torch.arange(0, num_nodes, dtype=torch.long,
                                device=edge_index.device)
        diag_idx = diag_idx.unsqueeze(0).repeat(2, 1)

        if normalization == 'sym':
            _, col = edge_index
            D_vec = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
            D_vec_invsqrt = 1 / torch.sqrt(D_vec)
            edge_index, edge_weight = spspmm(diag_idx, D_vec_invsqrt,
                                             edge_index, edge_weight,
                                             num_nodes, num_nodes, num_nodes)
            edge_index, edge_weight = spspmm(edge_index, edge_weight, diag_idx,
                                             D_vec_invsqrt, num_nodes,
                                             num_nodes, num_nodes)
        elif normalization == 'col':
            _, col = edge_index
            D_vec = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
            D_vec_inv = 1 / D_vec
            edge_index, edge_weight = spspmm(edge_index, edge_weight, diag_idx,
                                             D_vec_inv, num_nodes, num_nodes,
                                             num_nodes)
        elif normalization == 'row':
            row, _ = edge_index
            D_vec = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
            D_vec_inv = 1 / D_vec
            edge_index, edge_weight = spspmm(diag_idx, D_vec_inv, edge_index,
                                             edge_weight, num_nodes, num_nodes,
                                             num_nodes)
        elif normalization is None:
            pass
        else:
            raise ValueError(
                'Transition matrix normalization {} unknown.'.format(
                    normalization))

        return edge_index, edge_weight

    def diffusion_matrix_exact(self, edge_index, edge_weight, num_nodes,
                               method, **kwargs):
        r"""Calculate the (dense) diffusion on a given sparse graph.
        Note that these exact variants are not scalable. They densify the
        adjacency matrix and calculate either its inverse or its matrix
        exponential.

        Args:
            edge_index (LongTensor): The edge indices.
            edge_weight (Tensor): One-dimensional edge weights.
            num_nodes (int): Number of nodes.
            method (str): Diffusion method. Options:
                1. :obj:`"ppr"`: Use personalized PageRank as diffusion.
                Additionally expects the parameter:
                - alpha (float): Return probability in PPR. Commonly lies in
                    :obj:`[0.05, 0.2]`.

                2. `"heat"`: Use heat kernel diffusion.
                Additionally expects the parameter:
                - :obj:`t` (float): Time of diffusion. Commonly lies in
                    :obj:`[2, 10]`.

                3. :obj:`"coeff"`: Freely choose diffusion coefficients.
                Additionally expects the parameter:
                - coeffs (List[float]): List of coefficients :obj:`theta_k`
                for each power of the transition matrix (starting at :obj:`0`).

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
            raise ValueError('Exact GDC diffusion {} unknown.'.format(method))

        return diff_matrix

    def diffusion_matrix_approx(self, edge_index, edge_weight, num_nodes,
                                normalization, method, **kwargs):
        r"""Calculate the approximate, sparse diffusion on a given sparse
        graph.

        Args:
            edge_index (LongTensor): The edge indices.
            edge_weight (Tensor): One-dimensional edge weights.
            num_nodes (int): Number of nodes.
            normalization (str): Transition matrix normalization scheme
                (`"sym"`, `"row"`, or `"col"`).
                See :func:`GDC.transition_matrix` for details.
            method (str): Diffusion method. Options:
                1. `"ppr"`: Use personalized PageRank as diffusion.
                Additionally expects the parameters:
                - alpha (float): Return probability in PPR. Commonly lies in
                    :obj:`[0.05, 0.2]`.
                - eps (float): Threshold for PPR calculation stopping criterion
                    (:obj:`edge_weight >= eps * out_degree`).
                    Recommended default: :obj:`1e-4`.

        :rtype: (:class:`LongTensor`, :class:`Tensor`)
        """
        if method == 'ppr':
            if normalization == 'sym':
                # Calculate original degrees.
                _, col = edge_index
                D_vec_orig = scatter_add(edge_weight, col, dim=0,
                                         dim_size=num_nodes)

            edge_index_np = edge_index.cpu().numpy()
            # Assumes coalesced edge_index.
            _, indptr, out_degree = np.unique(edge_index_np[0],
                                              return_index=True,
                                              return_counts=True)

            neighbors, neighbor_weights = GDC.__calc_ppr__(
                indptr, edge_index_np[1], out_degree, kwargs['alpha'],
                kwargs['eps'])
            ppr_normalization = 'col' if normalization == 'col' else 'row'
            edge_index, edge_weight = self.__neighbors_to_graph__(
                neighbors, neighbor_weights, ppr_normalization,
                device=edge_index.device)

            if normalization == 'sym':
                # We can change the normalization from row-normalized to
                # symmetric by multiplying the resulting matrix with D^{1/2}
                # from the left and D^{-1/2} from the right.
                # Since we use the original degrees for this it will be like
                # we had used symmetric normalization from the beginning
                # (except for errors due to approximation).
                diag_idx = torch.arange(0, num_nodes, dtype=torch.long,
                                        device=edge_index.device)
                diag_idx = diag_idx.unsqueeze(0).repeat(2, 1)

                D_vec_sqrt = torch.sqrt(D_vec_orig)
                edge_index, edge_weight = spspmm(diag_idx, D_vec_sqrt,
                                                 edge_index, edge_weight,
                                                 num_nodes, num_nodes,
                                                 num_nodes)
                D_vec_invsqrt = 1 / D_vec_sqrt
                edge_index, edge_weight = spspmm(edge_index, edge_weight,
                                                 diag_idx, D_vec_invsqrt,
                                                 num_nodes, num_nodes,
                                                 num_nodes)
            elif normalization in ['col', 'row']:
                pass
            else:
                raise ValueError(
                    ('Transition matrix normalization {} not implemented for '
                     'non-exact GDC computation.').format(normalization))

        elif method == 'heat':
            raise NotImplementedError(
                ('Currently no fast heat kernel is implemented. You are '
                 'welcome to create one yourself, e.g., based on '
                 '"Kloster and Gleich: Heat kernel based community detection '
                 '(KDD 2014)."'))
        else:
            raise ValueError(
                'Approximate GDC diffusion {} unknown.'.format(method))

        return edge_index, edge_weight

    def sparsify_dense(self, matrix, method, **kwargs):
        r"""Sparsifies the given dense matrix.

        Args:
            matrix (Tensor): Matrix to sparsify.
            num_nodes (int): Number of nodes.
            method (str): Method of sparsification. Options:
                1. :obj:`"threshold"`: Remove all edges with weights smaller
                    than :obj:`eps`.
                    Additionally expects one of these parameters:
                - eps (float): Threshold to bound edges at.
                - avg_degree (int): If `eps` is not given,
                it can optionally be calculated by calculating
                the `eps` required to achieve a given `avg_degree`.

                2. `"topk"`: Keep edges with top `k` edge weights per node
                    (column).
                    Additionally expects the following parameters:
                    - k (int): Specifies the number of edges to keep.
                    - dim (int): The axis along which to take the top k.

        :rtype: (:class:`LongTensor`, :class:`Tensor`)
        """
        assert matrix.shape[0] == matrix.shape[1]
        N = matrix.shape[1]

        if method == 'threshold':
            if 'eps' not in kwargs.keys():
                kwargs['eps'] = self.__calculate_eps__(matrix, N,
                                                       kwargs['avg_degree'])

            edge_index = torch.nonzero(matrix >= kwargs['eps']).t()
            edge_index_flat = edge_index[0] * N + edge_index[1]
            edge_weight = matrix.flatten()[edge_index_flat]

        elif method == 'topk':
            assert kwargs['dim'] in [0, 1]
            sort_idx = torch.argsort(matrix, dim=kwargs['dim'],
                                     descending=True)
            if kwargs['dim'] == 0:
                top_idx = sort_idx[:kwargs['k']]
                edge_weight = torch.gather(matrix, dim=kwargs['dim'],
                                           index=top_idx).flatten()

                row_idx = torch.arange(0, N, device=matrix.device).repeat(
                    kwargs['k'])
                edge_index = torch.stack([top_idx.flatten(), row_idx], dim=0)
            else:
                top_idx = sort_idx[:, :kwargs['k']]
                edge_weight = torch.gather(matrix, dim=kwargs['dim'],
                                           index=top_idx).flatten()

                col_idx = torch.arange(
                    0, N, device=matrix.device).repeat_interleave(kwargs['k'])
                edge_index = torch.stack([col_idx, top_idx.flatten()], dim=0)
        else:
            raise ValueError('GDC sparsification {} unknown.'.format(method))

        return edge_index, edge_weight

    def sparsify_sparse(self, edge_index, edge_weight, num_nodes, method,
                        **kwargs):
        r"""Sparsifies a given sparse graph further.

        Args:
            edge_index (LongTensor): The edge indices.
            edge_weight (Tensor): One-dimensional edge weights.
            num_nodes (int): Number of nodes.
            method (str): Method of sparsification. Options:
                1. `"threshold"`: Remove all edges with weights smaller than
                :obj:`eps`.
                Additionally expects one of these parameters:
                - eps (float): Threshold to bound edges at.
                - avg_degree (int): If `eps` is not given,
                it can optionally be calculated by calculating
                the `eps` required to achieve a given `avg_degree`.

        :rtype: (:class:`LongTensor`, :class:`Tensor`)
        """
        if method == 'threshold':
            if 'eps' not in kwargs.keys():
                kwargs['eps'] = self.__calculate_eps__(edge_weight, num_nodes,
                                                       kwargs['avg_degree'])

            remaining_edge_idx = torch.nonzero(
                edge_weight >= kwargs['eps']).flatten()
            edge_index = edge_index[:, remaining_edge_idx]
            edge_weight = edge_weight[remaining_edge_idx]
        elif method == 'topk':
            raise NotImplementedError(
                'Sparse topk sparsification not implemented.')
        else:
            raise ValueError('GDC sparsification {} unknown.'.format(method))

        return edge_index, edge_weight

    def __expm__(self, matrix, symmetric):
        r"""Calculates matrix exponential.

        Args:
            matrix (Tensor): Matrix to take exponential of.
            symmetric (bool): Specifies whether the matrix is symmetric.

        :rtype: (:class:`Tensor`)
        """
        if symmetric:
            e, V = torch.symeig(matrix, eigenvectors=True)
            diff_mat = V @ torch.diag(e.exp()) @ V.t()
        else:
            diff_mat_np = expm(matrix.cpu().numpy())
            diff_mat = torch.Tensor(diff_mat_np).to(matrix.device)
        return diff_mat

    def __calculate_eps__(self, matrix, num_nodes, avg_degree):
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
        return sorted_edges[avg_degree * num_nodes - 1]

    def __neighbors_to_graph__(self, neighbors, neighbor_weights,
                               normalization='row', device='cpu'):
        r"""Combine a list of neighbors and neighbor weights to create a sparse
        graph.

        Args:
            neighbors (List[List[int]]): List of neighbors for each node.
            neighbor_weights (List[List[float]]): List of weights for the
                neighbors of each node.
            normalization (str): Normalization of resulting matrix
                (options: `row`, `col`). (default: :obj:`"row"`)
            device (torch.device): Device to create output tensors on.
                (default: :obj:`"cpu"`)

        :rtype: (:class:`LongTensor`, :class:`Tensor`)
        """
        edge_weight = torch.Tensor(np.concatenate(neighbor_weights)).to(device)
        i = np.repeat(np.arange(len(neighbors)),
                      np.fromiter(map(len, neighbors), dtype=np.int))
        j = np.concatenate(neighbors)
        if normalization == 'col':
            edge_index = torch.Tensor(np.vstack([j, i])).to(device)
            N = len(neighbors)
            edge_index, edge_weight = coalesce(edge_index, edge_weight, N, N)
        elif normalization == 'row':
            edge_index = torch.Tensor(np.vstack([i, j])).to(device)
        else:
            raise ValueError(
                f"PPR matrix normalization {normalization} unknown.")
        return edge_index, edge_weight

    @staticmethod
    @jit()
    def __calc_ppr__(indptr, indices, out_degree, alpha, eps):
        r"""Calculate the personalized PageRank vector for all nodes
        using a variant of the Andersen algorithm
        (see Andersen et al. :Local Graph Partitioning using PageRank Vectors.)

        Args:
            indptr (np.ndarray): Index pointer for the sparse matrix
                (CSR-format).
            indices (np.ndarray): Indices of the sparse matrix entries
                (CSR-format).
            out_degree (np.ndarray): Out-degree of each node.
            alpha (float): Alpha of the PageRank to calculate.
            eps (float): Threshold for PPR calculation stopping criterion
                (:obj:`edge_weight >= eps * out_degree`).

        :rtype: (:class:`List[List[int]]`, :class:`List[List[float]]`)
        """
        alpha_eps = alpha * eps
        js = []
        vals = []
        for inode in range(len(out_degree)):
            p = {inode: 0.0}
            r = {}
            r[inode] = alpha
            q = [inode]
            while len(q) > 0:
                unode = q.pop()

                res = r[unode] if unode in r else 0
                if unode in p:
                    p[unode] += res
                else:
                    p[unode] = res
                r[unode] = 0
                for vnode in indices[indptr[unode]:indptr[unode + 1]]:
                    _val = (1 - alpha) * res / out_degree[unode]
                    if vnode in r:
                        r[vnode] += _val
                    else:
                        r[vnode] = _val

                    res_vnode = r[vnode] if vnode in r else 0
                    if res_vnode >= alpha_eps * out_degree[vnode]:
                        if vnode not in q:
                            q.append(vnode)
            js.append(list(p.keys()))
            vals.append(list(p.values()))
        return js, vals

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
