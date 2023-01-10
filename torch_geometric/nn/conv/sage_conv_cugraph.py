import torch
import torch.nn.functional as F
from torch import nn
from torch_sparse import SparseTensor

from torch_geometric.utils import degree

try:
    from pylibcugraphops import make_fg_csr, make_mfg_csr
    from pylibcugraphops.torch.autograd import agg_concat_n2n as SAGEConvAgg
except ImportError:
    has_pylibcugraphops = False
else:
    has_pylibcugraphops = True


class SAGEConvCuGraph(nn.Module):
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

    def forward(self, x, edge_index, num_nodes, from_neighbor_sampler=False,
                max_num_neighbors=None):
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
