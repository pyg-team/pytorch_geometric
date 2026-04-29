from typing import Any, Optional

import torch
from torch import Tensor

from torch_geometric import EdgeIndex

try:  # pragma: no cover
    LEGACY_MODE = False
    from pylibcugraphops.pytorch import CSC, HeteroCSC
    HAS_PYLIBCUGRAPHOPS = True
except ImportError:
    HAS_PYLIBCUGRAPHOPS = False
    try:  # pragma: no cover
        from pylibcugraphops import (
            make_fg_csr,
            make_fg_csr_hg,
            make_mfg_csr,
            make_mfg_csr_hg,
        )
        LEGACY_MODE = True
    except ImportError:
        pass


class CuGraphModule(torch.nn.Module):  # pragma: no cover
    r"""An abstract base class for implementing :obj:`cugraph`-based message
    passing layers.
    """
    def __init__(self):
        super().__init__()

        if not HAS_PYLIBCUGRAPHOPS and not LEGACY_MODE:
            raise ModuleNotFoundError(f"'{self.__class__.__name__}' requires "
                                      f"'pylibcugraphops>=23.02'")

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""

    def get_cugraph(
        self,
        edge_index: EdgeIndex,
        max_num_neighbors: Optional[int] = None,
    ) -> Any:
        r"""Constructs a :obj:`cugraph` graph object from CSC representation.
        Supports both bipartite and non-bipartite graphs.

        Args:
            edge_index (EdgeIndex): The edge indices.
            max_num_neighbors (int, optional): The maximum number of neighbors
                of a target node. It is only effective when operating in a
                bipartite graph. When not given, will be computed on-the-fly,
                leading to slightly worse performance. (default: :obj:`None`)
        """
        if not isinstance(edge_index, EdgeIndex):
            raise ValueError(f"'edge_index' needs to be of type 'EdgeIndex' "
                             f"(got {type(edge_index)})")

        edge_index = edge_index.sort_by('col')[0]
        num_src_nodes = edge_index.get_sparse_size(0)
        (colptr, row), _ = edge_index.get_csc()

        if not row.is_cuda:
            raise RuntimeError(f"'{self.__class__.__name__}' requires GPU-"
                               f"based processing (got CPU tensor)")

        if num_src_nodes != colptr.numel() - 1:  # Bipartite graph:
            if max_num_neighbors is None:
                max_num_neighbors = int((colptr[1:] - colptr[:-1]).max())

            if LEGACY_MODE:
                dst_nodes = torch.arange(colptr.numel() - 1, device=row.device)
                return make_mfg_csr(dst_nodes, colptr, row, max_num_neighbors,
                                    num_src_nodes)

            return CSC(colptr, row, num_src_nodes,
                       dst_max_in_degree=max_num_neighbors)

        if LEGACY_MODE:
            return make_fg_csr(colptr, row)

        return CSC(colptr, row, num_src_nodes=num_src_nodes)

    def get_typed_cugraph(
        self,
        edge_index: EdgeIndex,
        edge_type: Tensor,
        num_edge_types: Optional[int] = None,
        max_num_neighbors: Optional[int] = None,
    ) -> Any:
        r"""Constructs a typed :obj:`cugraph` graph object from a CSC
        representation where each edge corresponds to a given edge type.
        Supports both bipartite and non-bipartite graphs.

        Args:
            edge_index (EdgeIndex): The edge indices.
            edge_type (torch.Tensor): The edge type.
            num_edge_types (int, optional): The maximum number of edge types.
                When not given, will be computed on-the-fly, leading to
                slightly worse performance. (default: :obj:`None`)
            max_num_neighbors (int, optional): The maximum number of neighbors
                of a target node. It is only effective when operating in a
                bipartite graph. When not given, will be computed on-the-fly,
                leading to slightly worse performance. (default: :obj:`None`)
        """
        if num_edge_types is None:
            num_edge_types = int(edge_type.max()) + 1

        if not isinstance(edge_index, EdgeIndex):
            raise ValueError(f"'edge_index' needs to be of type 'EdgeIndex' "
                             f"(got {type(edge_index)})")

        edge_index, perm = edge_index.sort_by('col')
        edge_type = edge_type[perm]
        num_src_nodes = edge_index.get_sparse_size(0)
        (colptr, row), _ = edge_index.get_csc()

        edge_type = edge_type.int()

        if num_src_nodes != colptr.numel() - 1:  # Bipartite graph:
            if max_num_neighbors is None:
                max_num_neighbors = int((colptr[1:] - colptr[:-1]).max())

            if LEGACY_MODE:
                dst_nodes = torch.arange(colptr.numel() - 1, device=row.device)
                return make_mfg_csr_hg(dst_nodes, colptr, row,
                                       max_num_neighbors, num_src_nodes,
                                       n_node_types=0,
                                       n_edge_types=num_edge_types,
                                       out_node_types=None, in_node_types=None,
                                       edge_types=edge_type)

            return HeteroCSC(colptr, row, edge_type, num_src_nodes,
                             num_edge_types,
                             dst_max_in_degree=max_num_neighbors)

        if LEGACY_MODE:
            return make_fg_csr_hg(colptr, row, n_node_types=0,
                                  n_edge_types=num_edge_types, node_types=None,
                                  edge_types=edge_type)

        return HeteroCSC(colptr, row, edge_type, num_src_nodes, num_edge_types)

    def forward(
        self,
        x: Tensor,
        edge_index: EdgeIndex,
        max_num_neighbors: Optional[int] = None,
    ) -> Tensor:
        r"""Runs the forward pass of the module.

        Args:
            x (torch.Tensor): The node features.
            edge_index (EdgeIndex): The edge indices.
            max_num_neighbors (int, optional): The maximum number of neighbors
                of a target node. It is only effective when operating in a
                bipartite graph. When not given, the value will be computed
                on-the-fly, leading to slightly worse performance.
                (default: :obj:`None`)
        """
        raise NotImplementedError
