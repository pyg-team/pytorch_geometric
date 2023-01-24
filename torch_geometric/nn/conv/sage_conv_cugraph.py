from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_sparse import SparseTensor

from torch_geometric.typing import Adj
from torch_geometric.utils import degree

try:
    from pylibcugraphops import make_fg_csr, make_mfg_csr
    from pylibcugraphops.torch.autograd import agg_concat_n2n as SAGEConvAgg
except ImportError:
    has_pylibcugraphops = False
else:
    has_pylibcugraphops = True


class SAGEConvCuGraph(nn.Module):
    r"""The GraphSAGE operator from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper, with the sparse
    aggregations accelerated by `cugraph-ops`.

    See :class:`SAGEConv` for the mathematical model.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        aggr (string, optional): The aggregation scheme to use (:obj:`"sum"`,
            :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`).
            (default: :obj:`"mean"`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        normalize (bool, optional): If set to :obj:`True`, output features
            will be :math:`\ell_2`-normalized, *i.e.*,
            :math:`\frac{\mathbf{x}^{\prime}_i}
            {\| \mathbf{x}^{\prime}_i \|_2}`.
            (default: :obj:`False`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)

    Shapes:
        - **inputs:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`(|\mathcal{V_s}|, F_{in})` if bipartite;
          edge indices :math:`(2, |\mathcal{E}|)`
        - **outputs:**
          node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V_t}|, F_{out})` if bipartite
    """
    def __init__(self, in_channels: int, out_channels: int, aggr: str = "mean",
                 normalize: bool = False, root_weight: bool = True,
                 bias: bool = True):
        if has_pylibcugraphops is False:
            raise ModuleNotFoundError(
                "torch_geometric.nn.SAGEConvCuGraph requires pylibcugraphops "
                ">= 23.02.00.")
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight
        self.aggr = aggr
        assert self.aggr in ['mean', 'sum', 'min', 'max']

        if self.root_weight:
            self.linear = nn.Linear(2 * in_channels, out_channels, bias=bias)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x: Tensor, edge_index: Adj,
                num_nodes: Optional[Union[int, Tuple[int, int]]] = None,
                max_num_neighbors: Optional[int] = None) -> Tensor:
        r"""
        Args:
            x (Tensor): The input node features.
            edge_index (LongTensor or SparseTensor): The edge indices.
            num_nodes (int or tuple, optional): The number of nodes in the
                graph. A tuple corresponds to (:obj:`num_src_nodes`,
                :obj:`num_trg_nodes`) for a bipartite graph. This field can
                only be omitted when :obj:`edge_index` is :obj:`SparseTensor`.
                (default: :obj:`None`)
            max_num_neighbors (int, optional): The maximum number of neighbors
                of a target node. It is only effective when :obj:`edge_index`
                is bipartite. For example, if the sampled graph is generated
                from :obj:`NeighborSampler`, the value should equal the
                :obj:`size` argument; if emitted from a :obj:`NeighborLoader`,
                the value should be set to :obj:`max(num_neighbors)`. When not
                given, the value will be computed on-the-fly using
                :obj:`degree()` utility, leading to slightly worse performance.
                (default: :obj:`None`)
        """
        _device = next(self.parameters()).device
        if _device.type != "cuda":
            raise RuntimeError(
                f"torch_geometric.nn.SAGEConvCuGraph requires the model to be "
                f"on device 'cuda', but got '{_device.type}'.")

        if isinstance(edge_index, SparseTensor):
            adj = edge_index
        else:
            assert num_nodes is not None
            if isinstance(num_nodes, int):
                num_nodes = (num_nodes, num_nodes)
            adj = SparseTensor(row=edge_index[0], col=edge_index[1],
                               sparse_sizes=num_nodes)

        # Create csc-representation.
        offsets, indices, _ = adj.csc()

        # Create cugraph-ops graph.
        if adj.size(0) != adj.size(1):
            if max_num_neighbors is None:
                assert isinstance(edge_index, Tensor)
                max_num_neighbors = int(degree(edge_index[1]).max().item())
            num_src_nodes = adj.size(0)
            dst_nodes = torch.arange(adj.size(1), device=_device)

            _graph = make_mfg_csr(dst_nodes, offsets, indices,
                                  max_num_neighbors, num_src_nodes)
        else:
            _graph = make_fg_csr(offsets, indices)

        out = SAGEConvAgg(x, _graph, self.aggr)
        if self.root_weight:
            out = self.linear(out)
        else:
            out = self.linear(out[:, :self.in_channels])

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, aggr={self.aggr})')
