from typing import Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch_sparse import SparseTensor

from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import degree

from ..inits import glorot, zeros

try:
    from pylibcugraphops import make_fg_csr_hg, make_mfg_csr_hg
    from pylibcugraphops.torch.autograd import \
        agg_hg_basis_n2n_post as RGCNConvAgg
except ImportError:
    has_pylibcugraphops = False
else:
    has_pylibcugraphops = True


class RGCNConvCuGraph(nn.Module):
    r"""The relational graph convolutional operator from the `"Modeling
    Relational Data with Graph Convolutional Networks"
    <https://arxiv.org/abs/1703.06103>`_ paper, with the sparse
    aggregations accelerated by `cugraph-ops`.

    See :class:`RGCNConv` for the mathematical model.

    Args:
        in_channels (int): Size of each input sample. In case no input features
            are given, this argument should correspond to the number of nodes
            in the graph.
        out_channels (int): Size of each output sample.
        num_relations (int): Number of relations.
        num_bases (int, optional): If set, this layer will use the
            basis-decomposition regularization scheme where :obj:`num_bases`
            denotes the number of bases to use. (default: :obj:`None`)
        aggr (string, optional): The aggregation scheme to use (:obj:`"add"`,
            :obj:`"sum"`, :obj:`"mean"`).
            (default: :obj:`"mean"`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)

    """
    def __init__(self, in_channels: int, out_channels: int, num_relations: int,
                 num_bases: Optional[int] = None, aggr: str = 'mean',
                 root_weight: bool = True, bias: bool = True):
        if has_pylibcugraphops is False:
            raise ModuleNotFoundError(
                "torch_geometric.nn.RGCNConvCuGraph requires pylibcugraphops "
                ">= 23.02.00.")
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.aggr = aggr
        self.root_weight = root_weight
        assert self.aggr in ['add', 'sum', 'mean']

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

    def forward(self, x: OptTensor, edge_index: Adj,
                edge_type: OptTensor = None,
                num_nodes: Optional[Union[int, Tuple[int, int]]] = None,
                max_num_neighbors: Optional[int] = None) -> Tensor:
        r"""
        Args:
            x: The input node features. Can be either a :obj:`[num_nodes,
                in_channels]` node feature matrix, or :obj:`None`. If set to
                :obj:`None`, the model uses an identity matrix for the node
                features.
            edge_index (LongTensor or SparseTensor): The edge indices.
            edge_type (Tensor, optional): The one-dimensional relation
                type/index for each edge in :obj:`edge_index`.
                Should be only :obj:`None` in case :obj:`edge_index` is of type
                :class:`torch_sparse.tensor.SparseTensor`.
                (default: :obj:`None`)
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
                f"torch_geometric.nn.RGCNConvCuGraph requires the model to be "
                f"on device 'cuda', but got '{_device.type}'.")

        if isinstance(edge_index, SparseTensor):
            adj = edge_index
            assert adj.storage.value() is not None
        else:
            assert edge_type is not None
            assert num_nodes is not None
            if isinstance(num_nodes, int):
                num_nodes = (num_nodes, num_nodes)
            adj = SparseTensor(row=edge_index[0], col=edge_index[1],
                               sparse_sizes=num_nodes, value=edge_type)

        # Create csc-representation.
        offsets, indices, edge_type_perm = adj.csc()
        edge_type_perm = edge_type_perm.int()

        # Create cugraph-ops graph.
        if adj.size(0) != adj.size(1):
            if max_num_neighbors is None:
                assert isinstance(edge_index, Tensor)
                max_num_neighbors = int(degree(edge_index[1]).max().item())
            num_src_nodes = adj.size(0)
            dst_nodes = torch.arange(adj.size(1), device=_device)

            _graph = make_mfg_csr_hg(dst_nodes, offsets, indices,
                                     max_num_neighbors, num_src_nodes,
                                     n_node_types=0,
                                     n_edge_types=self.num_relations,
                                     out_node_types=None, in_node_types=None,
                                     edge_types=edge_type_perm)
        else:
            _graph = make_fg_csr_hg(offsets, indices, n_node_types=0,
                                    n_edge_types=self.num_relations,
                                    node_types=None, edge_types=edge_type_perm)

        if x is None:
            x = torch.eye(self.in_channels, device=_device)

        out = RGCNConvAgg(x, self.comp, _graph, not self.root_weight,
                          bool(self.aggr == 'mean'))

        out = out @ self.weight.view(-1, self.out_channels)
        if self.bias is not None:
            out = out + self.bias

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, num_relations={self.num_relations})')
