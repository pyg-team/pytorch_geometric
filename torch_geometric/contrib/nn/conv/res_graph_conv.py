from typing import Optional

import torch
from torch import Tensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    SparseTensor,
    torch_sparse,
)
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils import add_self_loops as add_self_loops_fn
from torch_geometric.utils import (
    is_torch_sparse_tensor,
    scatter,
    spmm,
    to_edge_index,
)
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils.sparse import set_sparse_value


def res_gconv_norm(  # noqa: F811
    edge_index: Adj,
    edge_weight: OptTensor = None,
    num_nodes: Optional[int] = None,
    improved: bool = False,
    add_self_loops: bool = True,
    flow: str = "source_to_target",
    dtype: Optional[torch.dtype] = None,
):
    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        assert edge_index.size(0) == edge_index.size(1)

        adj_t = edge_index

        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = torch_sparse.fill_diag(adj_t, fill_value)

        # Implementation from https://github.com/jiechenjiechen/GNP
        # GNP/utils.py scale_A_by_spectral_radius(A)

        adj_t = adj_t.to_dense()
        absA = adj_t.abs()
        m, n = absA.shape
        row_sum = absA @ torch.ones(n, 1, dtype=absA.dtype, device=absA.device)
        col_sum = torch.ones(1, m, dtype=absA.dtype, device=absA.device) @ absA
        gamma = torch.min(torch.max(row_sum), torch.max(col_sum))

        adj_t = adj_t * (1. / gamma.item())
        adj_t = SparseTensor.from_dense(adj_t)
        return adj_t

    if is_torch_sparse_tensor(edge_index):
        assert edge_index.size(0) == edge_index.size(1)

        if edge_index.layout == torch.sparse_csc:
            raise NotImplementedError("Sparse CSC matrices are not yet "
                                      "supported in 'res_gconv_norm'")

        adj_t = edge_index
        if add_self_loops:
            adj_t, _ = add_self_loops_fn(adj_t, None, fill_value, num_nodes)

        edge_index, value = to_edge_index(adj_t)

        # Implementation from https://github.com/jiechenjiechen/GNP
        # GNP/utils.py scale_A_by_spectral_radius(A)

        absA = torch.absolute(adj_t)
        m, n = absA.shape
        row_sum = absA @ torch.ones(n, 1, dtype=adj_t.dtype,
                                    device=adj_t.device)
        col_sum = torch.ones(1, m, dtype=adj_t.dtype,
                             device=adj_t.device) @ absA
        gamma = torch.min(torch.max(row_sum), torch.max(col_sum))
        value = value / gamma

        return set_sparse_value(adj_t, value), None

    assert flow in ['source_to_target', 'target_to_source']
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if add_self_loops:
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                 device=edge_index.device)

    row, col = edge_index[0], edge_index[1]
    gamma_row = scatter(torch.abs(edge_weight), row, dim_size=num_nodes,
                        reduce='sum')
    gamma_col = scatter(torch.abs(edge_weight), col, dim_size=num_nodes,
                        reduce='sum')
    gamma = min(max(gamma_row), max(gamma_col))
    edge_weight = edge_weight / gamma

    return edge_index, edge_weight


class ResGConv(MessagePassing):
    r"""The graph convolutional operator with residual skip connections from
    the `"Graph Neural Preconditioners for Iterative Solutions of Sparse
    Linear Systems" <https://arxiv.org/pdf/2406.00809>`_ paper.

    .. math::
        \text{Res-GCONV}(\mathbf{X}) = \text{ReLU}\left(\mathbf{XU} +
        \mathbf{\hat{A}XW}\right)

    where
    :math:`\mathbf{\hat{A}}\in \mathbb{R}^{n\times n}` denotes the weighted
    adjacency matrix
    normalized using the normalization
    :math:`\mathbf{\hat{A}} = \mathbf{A}/\gamma`, where :math:`\gamma =
    \min\{\max_j\{\sum_j
    |a|_{ij}\},\max_i\{\sum_i |a|_{ij}\}\}`
    and :math:`\mathbf{W,U}` are matrices containing the learnable parameters.

    Args:
        channels (int): Size of each input and output sample.
            bias_u (bool, optional): If set to :obj:`False`, the layer will not
            learn an additive bias together with :math:`\mathbf{U}`.
            (default: obj:`True`)
        bias_u (bool, optional): If set to :obj:`False`, the layer will not
            learn an additive bias together with :math:`\mathbf{U}`.
            (default: obj:`True`)
        add_self_loops (bool, optional): If set to :obj:`True`, will add
            self-loops to the input graph. (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{A}} = \mathbf{A}/\gamma` on
            first execution, and will use the cached version for further
            executions. This parameter should only be set to :obj:`True` in
            transductive learning scenarios. (default: :obj:`False`)
        normalize (bool, optional): Whether to apply normalization
            by :math:`\gamma`. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F)`,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge weights :math:`(|\mathcal{E}|)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F)`
    """

    _cached_edge_index: Optional[OptPairTensor]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, channels: int, bias_u: bool = True, bias_w: bool = True,
                 add_self_loops: bool = False, cached: bool = False,
                 normalize: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.channels = channels
        self.cached = cached
        self.normalize = normalize
        self.add_self_loops = add_self_loops
        self.bias_u = bias_u
        self.bias_w = bias_w

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.U = Linear(self.channels, self.channels, bias=self.bias_u)
        self.W = Linear(self.channels, self.channels, bias=self.bias_w)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.U.reset_parameters()
        self.W.reset_parameters()
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = res_gconv_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim), False,
                        self.add_self_loops, self.flow, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = res_gconv_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim), False,
                        self.add_self_loops, self.flow, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        # the heart of the ResGConv layer
        # the linear layer, followed by the message passing
        # and the addition of the skip connection
        wx = self.W(x)
        awx = self.propagate(edge_index, x=wx, edge_weight=edge_weight)
        out = awx + self.U(x)

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.channels}, '
                f'bias_u={self.bias_u}, bias_w={self.bias_w})')
