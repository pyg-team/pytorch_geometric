from typing import List, Optional, Tuple, Union

import numba
import numpy as np
from torch import Tensor

from torch_geometric.utils import to_undirected


@numba.jit(nopython=True, parallel=True)
def _calc_ppr(
    indptr: np.ndarray,
    indices: np.ndarray,
    out_degree: np.ndarray,
    alpha: float,
    eps: float,
    target_nodes: Optional[np.ndarray] = None,
) -> Tuple[List[List[int]], List[List[float]]]:
    nnodes = len(out_degree) if target_nodes is None else len(target_nodes)
    alpha_eps = alpha * eps
    js = [[0]] * nnodes
    vals = [[0.]] * nnodes
    for inode_uint in numba.prange(nnodes):
        inode = numba.int64(inode_uint) if target_nodes is None \
            else target_nodes[inode_uint]
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
        js[inode_uint] = list(p.keys())
        vals[inode_uint] = list(p.values())
    return js, vals


def calculate_ppr(
    edge_index: Tensor,
    alpha: float = 0.2,
    eps: float = 1.e-5,
    num_nodes: Optional[int] = None,
    target_indices: Optional[Union[Tensor, np.ndarray]] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    r"""Calculate the personalized PageRank vector for selected nodes
        using a variant of the Andersen algorithm, given
        the edge_index of a graph.
        (see Andersen et al. :Local Graph Partitioning using PageRank Vectors.)

        Args:
            edge_index (torch.Tensor): Edge_index of a graph.
            alpha (float): Alpha of the PageRank to calculate.
            eps (float): Threshold for PPR calculation stopping criterion
                (:obj:`edge_weight >= eps * out_degree`).
            num_nodes (int, optional): Number of nodes.
            target_indices (torch.Tensor or np.ndarray, optional): Target
                nodes of PPR calculations. If not given, calculate for
                all the nodes by default.

        :rtype: (:class:`List[np.ndarray]`, :class:`List[np.ndarray]`)
        """
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    edge_index_np = edge_index.cpu().numpy()
    _, indptr, out_degree = np.unique(edge_index_np[0], return_index=True,
                                      return_counts=True)
    indptr = np.append(indptr, edge_index_np.shape[1])

    if isinstance(target_indices, Tensor):
        target_indices = target_indices.numpy()
    neighbors, weights = _calc_ppr(indptr, edge_index_np[1], out_degree, alpha,
                                   eps, target_indices)
    neighbors = [np.array(n) for n in neighbors]
    weights = [np.array(w) for w in weights]
    return neighbors, weights
