from typing import Any, Optional, Tuple

import torch
from torch import Tensor

from torch_geometric.utils import index_sort

try:
    from pylibcugraphops import make_fg_csr, make_mfg_csr
    HAS_PYLIBCUGRAPHOPS = True
except ImportError:
    HAS_PYLIBCUGRAPHOPS = False


class CuGraphModule(torch.nn.Module):  # pragma: no cover
    r"""An abstract base class for implementing cugraph message passing layers.
    """
    def __init__(self):
        super().__init__()

        if HAS_PYLIBCUGRAPHOPS is False:
            raise ModuleNotFoundError(f"'{self.__class__.__name__}' requires "
                                      f"'pylibcugraphops'")

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        pass

    @staticmethod
    def to_csc(
        edge_index: Tensor,
        size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[Tensor, Tensor]:
        r"""Returns a CSC representation of an :obj:`edge_index` tensor to be
        used as input to a :class:`CuGraphModule`.

        Args:
            edge_index (torch.Tensor): The edge indices.
            size ((int, int), optional). The shape of :obj:`edge_index` in each
                dimension. (default: :obj:`None`)
        """
        row, col = edge_index
        num_target_nodes = size[1] if size is not None else int(col.max() + 1)
        col, perm = index_sort(col, max_value=num_target_nodes)
        row = row[perm]

        colptr = torch._convert_indices_from_coo_to_csr(col, num_target_nodes)

        return row, colptr

    def get_cugraph(self, num_src_nodes: int, csc: Tuple[Tensor, Tensor],
                    max_num_neighbors: Optional[int] = None) -> Any:
        r"""Constructs a :obj:`cugraph` graph object from CSC representation.
        Supports both bipartite and non-bipartite graphs.

        Args:
            num_src_nodes (int): The number of source nodes.
            csc ((torch.Tensor, torch.Tensor)): A tuple containing the CSC
                representation of a graph, given as a tuple of
                :obj:`(row, colptr)`. Use the :meth:`to_csc` method to convert
                an :obj:`edge_index` representation to the desired format.
            max_num_neighbors (int, optional): The maximum number of neighbors
                of a target node. It is only effective when operating in a
                bipartite graph.. When not given, the value will be computed
                on-the-fly, leading to slightly worse performance.
                (default: :obj:`None`)
        """
        row, colptr = csc

        if not row.is_cuda:
            raise RuntimeError(f"'{self.__class__.__name__}' requires GPU-"
                               f"based processing (got CPU tensor)")

        if num_src_nodes != colptr.numel() - 1:  # Bipartite graph:
            if max_num_neighbors is None:
                max_num_neighbors = int((colptr[1:] - colptr[:-1]).max())

            dst_nodes = torch.arange(colptr.numel() - 1, device=row.device)

            return make_mfg_csr(dst_nodes, colptr, row, max_num_neighbors,
                                num_src_nodes)

        return make_fg_csr(colptr, row)

    def forward(
        self,
        x: Tensor,
        csc: Tuple[Tensor, Tensor],
        max_num_neighbors: Optional[int] = None,
    ) -> Tensor:
        r"""
        Args:
            x (torch.Tensor): The node features.
            csc ((torch.Tensor, torch.Tensor)): A tuple containing the CSC
                representation of a graph, given as a tuple of
                :obj:`(row, colptr)`. Use the :meth:`to_csc` method to convert
                an :obj:`edge_index` representation to the desired format.
            max_num_neighbors (int, optional): The maximum number of neighbors
                of a target node. It is only effective when operating in a
                bipartite graph.. When not given, the value will be computed
                on-the-fly, leading to slightly worse performance.
                (default: :obj:`None`)
        """
        raise NotImplementedError
