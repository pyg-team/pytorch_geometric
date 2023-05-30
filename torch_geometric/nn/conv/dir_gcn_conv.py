from typing import Optional

import torch
from torch import Tensor

from torch_geometric.nn.conv import GCNConv
from torch_geometric.typing import Adj, SparseTensor, torch_sparse
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils import add_self_loops as add_self_loops_fn
from torch_geometric.utils import (
    is_torch_sparse_tensor,
    scatter,
    to_edge_index,
)
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils.sparse import set_sparse_value

# from torch.nn import Parameter


@torch.jit._overload
def dir_gcn_norm(edge_index, edge_weight, num_nodes, add_self_loops, flow,
                 dtype):
    # type: (Tensor, OptTensor, Optional[int], bool, str, Optional[int]) -> OptPairTensor  # noqa
    pass


@torch.jit._overload
def dir_gcn_norm(edge_index, edge_weight, num_nodes, add_self_loops, flow,
                 dtype):
    # type: (SparseTensor, OptTensor, Optional[int], bool, str, Optional[int]) -> SparseTensor  # noqa
    pass


def dir_gcn_norm(edge_index, edge_weight=None, num_nodes=None,
                 add_self_loops=False, flow="source_to_target", dtype=None):

    if isinstance(edge_index, SparseTensor):
        assert edge_index.size(0) == edge_index.size(1)

        adj_t = edge_index
        adj = edge_index.t()

        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = torch_sparse.fill_diag(adj_t, 1.)

        deg = torch_sparse.sum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = torch_sparse.mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = torch_sparse.mul(adj_t, deg_inv_sqrt.view(1, -1))

        in_deg = torch_sparse.sum(adj, dim=0)
        in_deg_inv_sqrt = in_deg.pow_(-0.5)
        in_deg_inv_sqrt.masked_fill_(in_deg_inv_sqrt == float("inf"), 0.0)

        out_deg = torch_sparse.sum(adj, dim=1)
        out_deg_inv_sqrt = out_deg.pow_(-0.5)
        out_deg_inv_sqrt.masked_fill_(out_deg_inv_sqrt == float("inf"), 0.0)

        adj = torch_sparse.mul(adj, out_deg_inv_sqrt.view(-1, 1))
        adj = torch_sparse.mul(adj, in_deg_inv_sqrt.view(1, -1))

        return adj_t

    if is_torch_sparse_tensor(edge_index):
        assert edge_index.size(0) == edge_index.size(1)

        if edge_index.layout == torch.sparse_csc:
            raise NotImplementedError("Sparse CSC matrices are not yet "
                                      "supported in 'gcn_norm'")

        adj_t = edge_index
        if add_self_loops:
            adj_t, _ = add_self_loops_fn(adj_t, None, 1., num_nodes)

        edge_index, value = to_edge_index(adj_t)
        col, row = edge_index[0], edge_index[1]

        in_deg = scatter(value, col, 0, dim_size=num_nodes, reduce='sum')
        in_deg_inv_sqrt = in_deg.pow_(-0.5)
        in_deg_inv_sqrt.masked_fill_(in_deg_inv_sqrt == float('inf'), 0)

        out_deg = scatter(value, col, 0, dim_size=num_nodes, reduce='sum')
        out_deg_inv_sqrt = out_deg.pow_(-0.5)
        out_deg_inv_sqrt.masked_fill_(out_deg_inv_sqrt == float('inf'), 0)

        value = out_deg_inv_sqrt[row] * value * in_deg_inv_sqrt[col]

        return set_sparse_value(adj_t, value), None

    assert flow in ['source_to_target', 'target_to_source']
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if add_self_loops:
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, 1., num_nodes)

    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                 device=edge_index.device)

    row, col = edge_index[0], edge_index[1]
    # idx = col if flow == 'source_to_target' else row

    in_deg = scatter(edge_weight, col, dim=0, dim_size=num_nodes, reduce='sum')
    in_deg_inv_sqrt = in_deg.pow_(-0.5)
    in_deg_inv_sqrt.masked_fill_(in_deg_inv_sqrt == float('inf'), 0)

    out_deg = scatter(edge_weight, row, dim=0, dim_size=num_nodes,
                      reduce='sum')
    out_deg_inv_sqrt = out_deg.pow_(-0.5)
    out_deg_inv_sqrt.masked_fill_(out_deg_inv_sqrt == float('inf'), 0)

    edge_weight = out_deg_inv_sqrt[row] * edge_weight * in_deg_inv_sqrt[col]

    return edge_index, edge_weight


class DirGCNConv(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 alpha: Optional[float] = 0.5, cached=True,
                 add_self_loops=False, **kwargs):
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_src_to_dst = GCNConv(in_channels, out_channels,
                                       add_self_loops=False, normalize=False)
        self.conv_dst_to_src = GCNConv(in_channels, out_channels,
                                       add_self_loops=False, normalize=False)
        self.alpha = alpha
        self.cached = cached
        self.add_self_loops = add_self_loops

        self._cached_edge_index = None
        self._cached_edge_index_t = None

        self.reset_parameters()

    def reset_parameters(self):
        self.conv_src_to_dst.reset_parameters()
        self.conv_dst_to_src.reset_parameters()

    def get_norm_edge_index(self, edge_index, cache, x):
        if cache is None:
            edge_index, edge_weight = dir_gcn_norm(  # yapf: disable
                edge_index, None, x.size(0), self.add_self_loops,
                "source_to_target", x.dtype)
        else:
            edge_index, edge_weight = cache[0], cache[1]

        return edge_index, edge_weight

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        edge_index, edge_weight = self.get_norm_edge_index(
            edge_index, self._cached_edge_index, x)
        edge_index_t, edge_weight_t = self.get_norm_edge_index(
            edge_index.flip(0), self._cached_edge_index_t, x)

        if self.cached:
            self._cached_edge_index = edge_index, edge_weight
            self._cached_edge_index_t = edge_index_t, edge_weight_t

        x_in = self.conv_dst_to_src(x, edge_index_t, edge_weight_t)
        x_out = self.conv_src_to_dst(x, edge_index, edge_weight)

        return self.alpha * x_in + (1 - self.alpha) * x_out

    def __repr__(self) -> str:
        if hasattr(self, 'in_channels') and hasattr(self, 'out_channels'):
            return (f'{self.__class__.__name__}({self.in_channels}, '
                    f'{self.out_channels})')
        return f'{self.__class__.__name__}()'
