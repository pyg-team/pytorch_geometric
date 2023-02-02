from typing import Optional, Tuple

import torch
from torch import Tensor

from torch_geometric.utils import index_sort


class CuGraphModule(torch.nn.Module):  # pragma: no cover
    r"""An abstract base class for implementing cugraph message passing layers.
    """
    @staticmethod
    def to_csc(
        edge_index: Tensor,
        size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[Tensor, Tensor]:
        r"""
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
