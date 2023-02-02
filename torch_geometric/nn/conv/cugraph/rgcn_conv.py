from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from torch_geometric.nn.conv.cugraph import CuGraphModule
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import OptTensor

try:
    from pylibcugraphops import make_fg_csr_hg, make_mfg_csr_hg
    from pylibcugraphops.torch.autograd import \
        agg_hg_basis_n2n_post as RGCNConvAgg
    HAS_PYLIBCUGRAPHOPS = True
except ImportError:
    HAS_PYLIBCUGRAPHOPS = False


class CuGraphRGCNConv(CuGraphModule):  # pragma: no cover
    r"""The relational graph convolutional operator from the `"Modeling
    Relational Data with Graph Convolutional Networks"
    <https://arxiv.org/abs/1703.06103>`_ paper.

    :class:`CuGraphRGCNConv` is an optimized version of
    :class:`~torch_geometric.nn.conv.RGCNConv` based on the :obj:`cugraph-ops`
    package that fuses message passing computation for accelerated execution
    and lower memory footprint.
    """
    def __init__(self, in_channels: int, out_channels: int, num_relations: int,
                 num_bases: Optional[int] = None, aggr: str = 'mean',
                 root_weight: bool = True, bias: bool = True):
        if HAS_PYLIBCUGRAPHOPS is False:
            raise ModuleNotFoundError(f"'{self.__class__.__name__}' requires "
                                      f"'pylibcugraphops >= 23.02.00'")

        if aggr not in ['add', 'sum', 'mean']:
            raise ValueError(f"Aggregation function must be either 'mean', "
                             f"'add', 'sum' or 'mean' (got '{aggr}')")

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.aggr = aggr
        self.root_weight = root_weight

        dim_root_weight = 1 if root_weight else 0

        if num_bases is not None:
            self.weight = nn.Parameter(
                torch.Tensor(num_bases + dim_root_weight, in_channels,
                             out_channels))
            self.comp = nn.Parameter(torch.Tensor(num_relations, num_bases))
        else:
            self.weight = nn.Parameter(
                torch.Tensor(num_relations + dim_root_weight, in_channels,
                             out_channels))
            self.comp = None

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        end = -1 if self.root_weight else None
        glorot(self.weight[:end])
        glorot(self.comp)
        if self.root_weight:
            glorot(self.weight[-1])
        zeros(self.bias)

    def forward(self, x: OptTensor, csc: Tuple[Tensor, Tensor, Tensor],
                max_num_neighbors: Optional[int] = None) -> Tensor:
        if x is None:
            x = torch.eye(self.in_channels, device=csc.device)

        if not x.is_cuda:
            raise RuntimeError(f"'{self.__class__.__name__}' requires GPU-"
                               f"based processing (got CPU tensor)")

        row, colptr, edge_type = csc
        edge_type = edge_type.int()

        # Create csc-representation.
        if x.size(0) != colptr.numel() - 1:  # Operating in a bipartite graph:
            if max_num_neighbors is None:
                max_num_neighbors = int((colptr[1:] - colptr[:-1]).max())

            num_src_nodes = x.size(0)
            dst_nodes = torch.arange(colptr.numel() - 1, device=x.device)

            _graph = make_mfg_csr_hg(dst_nodes, colptr, row, max_num_neighbors,
                                     num_src_nodes, n_node_types=0,
                                     n_edge_types=self.num_relations,
                                     out_node_types=None, in_node_types=None,
                                     edge_types=edge_type)
        else:
            _graph = make_fg_csr_hg(colptr, row, n_node_types=0,
                                    n_edge_types=self.num_relations,
                                    node_types=None, edge_types=edge_type)

        out = RGCNConvAgg(x, self.comp, _graph, concat_own=self.root_weight,
                          norm_by_out_degree=bool(self.aggr == 'mean'))

        out = out @ self.weight.view(-1, self.out_channels)

        if self.bias is not None:
            out = out + self.bias

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, num_relations={self.num_relations})')
