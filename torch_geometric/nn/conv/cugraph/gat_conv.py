from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from torch_geometric.nn.conv.cugraph import CuGraphModule
from torch_geometric.nn.inits import zeros

try:
    from pylibcugraphops import make_fg_csr, make_mfg_csr
    from pylibcugraphops.torch.autograd import mha_gat_n2n as GATConvAgg
    HAS_PYLIBCUGRAPHOPS = True
except ImportError:
    HAS_PYLIBCUGRAPHOPS = False


class CuGraphGATConv(CuGraphModule):  # pragma: no cover
    r"""The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper.

    :class:`CuGraphGATConv` is an optimized version of
    :class:`~torch_geometric.nn.conv.GATConv` based on the :obj:`cugraph-ops`
    package that fuses message passing computation for accelerated execution
    and lower memory footprint.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        bias: bool = True,
    ):
        if HAS_PYLIBCUGRAPHOPS is False:
            raise ModuleNotFoundError(f"'{self.__class__.__name__}' requires "
                                      f"'pylibcugraphops >= 23.02.00'")

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope

        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.att = nn.Parameter(torch.Tensor(2 * heads * out_channels))

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.att.view(2, self.heads, self.out_channels),
                               gain=gain)
        zeros(self.bias)

    def forward(
        self,
        x: Tensor,
        csc: Tuple[Tensor, Tensor],
        max_num_neighbors: Optional[int] = None,
    ) -> Tensor:
        if not x.is_cuda:
            raise RuntimeError(f"'{self.__class__.__name__}' requires GPU-"
                               f"based processing (got CPU tensor)")

        row, colptr = csc

        # Create `cugraph-ops`` graph:
        if x.size(0) != colptr.numel() - 1:  # Operating in a bipartite graph:
            if max_num_neighbors is None:
                max_num_neighbors = int((colptr[1:] - colptr[:-1]).max())

            num_src_nodes = x.size(0)
            dst_nodes = torch.arange(colptr.numel() - 1, device=x.device)

            _graph = make_mfg_csr(dst_nodes, colptr, row, max_num_neighbors,
                                  num_src_nodes)
        else:
            _graph = make_fg_csr(colptr, row)

        x = self.lin(x)
        out = GATConvAgg(x, self.att, _graph, self.heads, "LeakyReLU",
                         self.negative_slope, False, self.concat)

        if self.bias is not None:
            out = out + self.bias

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')
