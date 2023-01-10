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

    def forward(self, x: Tensor, edge_index: Adj, num_nodes: int,
                from_neighbor_sampler: bool = False,
                max_num_neighbors: int = None):
        r"""
        Args:
            x: The input node features.
            edge_index (LongTensor or SparseTensor): The edge indices.
            num_nodes (int): The number of nodes in the graph.
            from_neighbor_sampler: Set to :obj:`True` when :obj:`edge_index`
                comes from a neighbor sampler. This allows the model to opt for
                a more performant aggregation primitive that is designed for
                sampled graphs. (default: :obj:`False`)
            max_num_neighbors (int, optional): The maximum number of neighbors
                of an output node, i.e. :obj:`max(num_neighbors)`, with
                the :obj:`num_neighbors` being the one used in the neighbor
                sampler. It only becomes effective when
                :obj:`from_neighbor_sampler` is :obj:`True`. If :obj:`None`,
                the value will be computed on-the-fly using the :obj:`degree()`
                utility, and therefore, it is recommended to pass in this value
                directly for better performance. (default: :obj:`None`)
        """
        _device = next(self.parameters()).device
        if _device.type != "cuda":
            raise RuntimeError(
                f"torch_geometric.nn.SAGEConvCuGraph requires the model to be "
                f"on device 'cuda', but got '{_device.type}'.")

        # Create csc-representation and cast edge_types to int32.
        if isinstance(edge_index, SparseTensor):
            adj = edge_index
            assert adj.storage.value() is not None
            assert adj.size(0) == adj.size(1)
        else:
            adj = SparseTensor(row=edge_index[0], col=edge_index[1],
                               sparse_sizes=(num_nodes, num_nodes))

        offsets, indices, _ = adj.csc()

        # Create cugraph-ops graph.
        if from_neighbor_sampler:
            if max_num_neighbors is None:
                max_num_neighbors = int(degree(edge_index[1]).max().item())

            src_nodes = torch.arange(num_nodes, device=_device)
            dst_nodes = torch.arange(num_nodes, device=_device)

            _graph = make_mfg_csr(dst_nodes, src_nodes, offsets, indices,
                                  max_num_neighbors)
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
