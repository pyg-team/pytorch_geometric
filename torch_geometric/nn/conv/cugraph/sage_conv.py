from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear

from torch_geometric.utils import index_sort

try:
    from pylibcugraphops import make_fg_csr, make_mfg_csr
    from pylibcugraphops.torch.autograd import agg_concat_n2n as SAGEConvAgg
    HAS_PYLIBCUGRAPHOPS = True
except ImportError:
    HAS_PYLIBCUGRAPHOPS = False


class CuGraphSAGEConv(torch.nn.Module):
    r"""The GraphSAGE operator from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper.

    :class:`CuGraphSAGEConv` is an optimized version of
    :class:`~torch_geometric.nn.conv.SAGEConv` based on the :obj:`cugraph-ops`
    package that fuses message passing computation for accelerated exeuction
    and lower memory footprint.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aggr: str = 'mean',
        normalize: bool = False,
        root_weight: bool = True,
        project: bool = False,
        bias: bool = True,
    ):
        if HAS_PYLIBCUGRAPHOPS is False:
            raise ModuleNotFoundError(f"'{self.__class__.__name__}' requires "
                                      f"'pylibcugraphops >= 23.02.00'")

        if aggr not in ['mean', 'sum', 'min', 'max']:
            raise ValueError(f"Aggregation function must be either 'mean', "
                             f"'sum', 'min' or 'max' (got '{aggr}')")

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggr = aggr
        self.normalize = normalize
        self.root_weight = root_weight
        self.project = project

        if self.project:
            self.pre_lin = Linear(in_channels, in_channels, bias=True)

        if self.root_weight:
            self.lin = Linear(2 * in_channels, out_channels, bias=bias)
        else:
            self.lin = Linear(in_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        if self.project:
            self.pre_lin.reset_parameters()
        self.linear.reset_parameters()

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
        # TODO Consider out-sourcing to base class or utility function.
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
        if not x.is_cuda:
            raise RuntimeError(f"'{self.__class__.__name__}' requires GPU-"
                               f"based processing (got CPU tensor)")

        if self.project:
            x = self.pre_lin(x).relu()

        row, colptr = csc

        # Create `cugraph-ops` graph:
        if x.size(0) != colptr.numel() - 1:  # Operating in a bipartite graph:
            if max_num_neighbors is None:
                max_num_neighbors = int((colptr[1:] - colptr[:-1]).max())

            num_src_nodes = x.size(0)
            dst_nodes = torch.arange(colptr.numel() - 1, device=x.device)

            _graph = make_mfg_csr(dst_nodes, colptr, row, max_num_neighbors,
                                  num_src_nodes)
        else:
            _graph = make_fg_csr(colptr, row)

        out = SAGEConvAgg(x, _graph, self.aggr)

        if self.root_weight:
            out = self.lin(out)
        else:
            out = self.lin(out[:, :self.in_channels])

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, aggr={self.aggr})')
