import torch
from torch import Tensor, nn
from torch_sparse import SparseTensor, set_diag

from torch_geometric.typing import Adj
from torch_geometric.utils import add_self_loops, degree

from ..inits import zeros

try:
    from pylibcugraphops import make_fg_csr, make_mfg_csr
    from pylibcugraphops.torch.autograd import mha_gat_n2n as GATConvAgg
except ImportError:
    has_pylibcugraphops = False
else:
    has_pylibcugraphops = True


class GATConvCuGraph(nn.Module):
    r"""The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper, with the sparse aggregations
    accelerated by `cugraph-ops`.

    See :class:`GATConv` for the mathematical model.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        add_self_loops: bool = True,
        bias: bool = True,
    ):

        if has_pylibcugraphops is False:
            raise ModuleNotFoundError(
                "torch_geometric.nn.GATConvCuGraph requires pylibcugraphops "
                ">= 23.02.00.")
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.add_self_loops = add_self_loops

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
                f"torch_geometric.nn.GATConvCuGraph requires the model to be "
                f"on device 'cuda', but got '{_device.type}'.")

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                edge_index, _ = add_self_loops(edge_index)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        # Create csc-representation.
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

        x = self.lin(x)
        out = GATConvAgg(x, self.att, _graph, self.heads, "LeakyReLU",
                         self.negative_slope, False, self.concat)

        if self.bias is not None:
            out = out + self.bias

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')
