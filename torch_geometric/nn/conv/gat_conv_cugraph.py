from typing import Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch_sparse import SparseTensor, set_diag

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

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`(|\mathcal{V_s}|, F_{in})` if bipartite;
          edge indices :math:`(2, |\mathcal{E}|)`,
        - **output:** node features :math:`(|\mathcal{V}|, H * F_{out})` or
          :math:`((|\mathcal{V}_t|, H * F_{out})` if bipartite.
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

    def forward(self, x: Tensor, edge_index: Adj,
                num_nodes: Optional[Union[int, Tuple[int, int]]] = None,
                max_num_neighbors: Optional[int] = None) -> Tensor:
        r"""
        Args:
            x: The input node features.
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
                f"torch_geometric.nn.GATConvCuGraph requires the model to be "
                f"on device 'cuda', but got '{_device.type}'.")

        if isinstance(edge_index, SparseTensor):
            adj = edge_index
        else:
            assert num_nodes is not None
            if isinstance(num_nodes, int):
                num_nodes = (num_nodes, num_nodes)
            adj = SparseTensor(row=edge_index[0], col=edge_index[1],
                               sparse_sizes=num_nodes)
        if self.add_self_loops:
            adj = set_diag(adj)

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

        x = self.lin(x)
        out = GATConvAgg(x, self.att, _graph, self.heads, "LeakyReLU",
                         self.negative_slope, False, self.concat)

        if self.bias is not None:
            out = out + self.bias

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')
