from typing import Optional

import torch
from torch import nn
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
    This module depends on :code:`pylibcugraphops` package, which can be
    installed via :code:`conda install -c nvidia pylibcugraphops>=23.02`.
    .. note::
        Compared with :class:`torch_geometric.nn.conv.RGCNConv`, this model:
        * Only works on cuda devices.
        * Only supports basis-decomposition regularization.
    Args:
        in_channels (int): Size of each input sample.
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
        num_neighbors (int, optional): The maximum number of neighbors of an
            output node. It is recommended to pass in the value for better
            performance. It should be the same as the `num_neighbors` used in
            :class:`torch_geometric.loader.NeighborLoader`. If :obj:`None`,
            the value will be computed on-the-fly. (default: :obj:`None`)
    """
    def __init__(self, in_channels: int, out_channels: int, num_relations: int,
                 num_bases: Optional[int] = None, aggr: str = 'mean',
                 root_weight: bool = True, bias: bool = True,
                 num_neighbors: Optional[int] = None):
        if has_pylibcugraphops is False:
            raise ModuleNotFoundError(
                "torch_geometric.nn.RGCNConvCuGraph requires pylibcugraphops "
                "to be installed.")
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.num_neighbors = num_neighbors
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
        glorot(self.weight)
        glorot(self.comp)
        zeros(self.bias)

    def forward(self, x: OptTensor, edge_index: Adj,
                edge_type: OptTensor = None,
                from_neighbor_sampler: Optional[bool] = False):
        r"""
        Args:
            x: The input node features. Can be either a :obj:`[num_nodes,
                in_channels]` node feature matrix, or :obj:`None`. If set to
                :obj:`None`, the model uses an identity matrix for the node
                features.
            edge_index (LongTensor or SparseTensor): The edge indices.
            edge_type: The one-dimensional relation type/index for each edge in
                :obj:`edge_index`.
                Should be only :obj:`None` in case :obj:`edge_index` is of type
                :class:`torch_sparse.tensor.SparseTensor`.
                (default: :obj:`None`)
            from_neighbor_sampler: Set to :obj:`True` when :obj:`edge_index`
                comes from a neighbor sampler. This allows the model to opt for
                a more performant aggregation primitive.
                (default: :obj:`False`)
        """
        _device = next(self.parameters()).device
        if _device.type != "cuda":
            raise RuntimeError(
                f"torch_geometric.nn.RGCNConvCuGraph requires the model to be "
                f"on device 'cuda', but got '{_device.type}'.")

        # Create csc-representation and cast edge_types to int32.
        if isinstance(edge_index, SparseTensor):
            adj = edge_index
            assert adj.storage.value() is not None
        else:
            assert edge_type is not None
            adj = SparseTensor(row=edge_index[0], col=edge_index[1],
                               value=edge_type)

        offsets, indices, edge_type_perm = adj.csc()
        edge_type_perm = edge_type_perm.int()
        num_src_nodes, num_dst_nodes = adj.sparse_sizes()

        # Create cugraph-ops graph.
        if from_neighbor_sampler:
            num_neighbors = self.num_neighbors
            if num_neighbors is None:
                num_neighbors = int(degree(edge_index[1]).max().item())

            src_nodes = torch.arange(num_src_nodes, device=_device)
            dst_nodes = torch.arange(num_dst_nodes, device=_device)

            _graph = make_mfg_csr_hg(dst_nodes, src_nodes, offsets, indices,
                                     num_neighbors, n_node_types=0,
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
