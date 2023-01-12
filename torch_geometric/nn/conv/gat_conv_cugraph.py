import torch
from torch import Tensor, nn
from torch_sparse import SparseTensor

from torch_geometric.typing import Adj
from torch_geometric.utils import degree

from ..inits import zeros

try:
    from pylibcugraphops import make_fg_csr, make_mfg_csr
    from pylibcugraphops.torch.autograd import mha_gat_n2n as GATConvAgg
except ImportError:
    has_pylibcugraphops = False
else:
    has_pylibcugraphops = True


class GATConvCuGraph(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        bias: bool = True,
    ):
        if has_pylibcugraphops is False:
            raise ModuleNotFoundError(
                "torch_geometric.nn.SAGEConvCuGraph requires pylibcugraphops "
                ">= 23.02.00.")
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
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

        x = self.lin(x)
        out = GATConvAgg(x, self.att, _graph, self.heads, "LeakyReLU",
                         self.negative_slope, False, self.concat)

        if self.bias is not None:
            out = out + self.bias

        return out
