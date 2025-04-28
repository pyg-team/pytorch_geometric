from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import zeros
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


def block_diagonal_init(weight_matrix, block_size=2, bound=0.5):
    # Get the size of the weight matrix
    n = weight_matrix.size(0)

    weight_matrix = torch.zeros_like(weight_matrix)

    # Calculate the number of blocks along one dimension
    num_blocks = (n + block_size - 1) // block_size

    for i in range(num_blocks):
        # Calculate the actual block size for the current block
        actual_block_size = min(block_size, n - i * block_size)

        # Initialize with Gaussian noise
        part = torch.randn(actual_block_size, actual_block_size) * bound

        # Place the block on the diagonal of the weight matrix
        start_row = i * block_size
        end_row = start_row + actual_block_size
        weight_matrix[start_row:end_row, start_row:end_row] = part

    return weight_matrix


class OrthoConv(nn.Module):
    r"""OrthoConv Layer.

    The **OrthoConv Layer** implements an equivariant graph convolution
    that is orthogonal (i.e., norm preserving and invertible) as described
    in the `“Unitary Convolutions for Learning on Graphs and Groups”
    <https://arxiv.org/abs/2410.05499>`_ paper. The layer is designed to
    work entirely with real numbers by pairing components to represent
    complex values.

    There are two variants:

    * **OrthoConv** (`use_lie=False`):

      An adjacency matrix :math:`A \in \mathbb{R}^{n \times n}` and
      a feature matrix :math:`X \in \mathbb{R}^{n \times d}`
      are used along with a trainable
      weight matrix :math:`W \in \mathbb{R}^{d \times d'}` and
      bias :math:`b \in \mathbb{R}^{d'}`. The operator is defined as

      .. math::
         f_{Uconv}(X) = \exp(iA) \, XW + b,

      where the matrix exponential is computed over :math:`iA` and
      normalization of :math:`A` is handled internally.

      **Important:** The output dimension must be even since real
      numbers are paired to represent complex numbers.

    * **Lie OrthoConv** (`use_lie=True`):

      The transformation is constrained to the Lie algebra of
      the orthogonal group. Here, the input and output dimensions
      must match and the weight matrix is skew-symmetric:

      .. math::
         f_{Uconv}(X) = \exp(g_{conv}(X)) + b,
         \quad \text{with} \quad g_{conv}(X) = AXW,
         \quad \text{and} \quad W + W^\top = 0.

      **Important:** In this variant, the input dimension must equal
      the output dimension.

    Args:
        dim_in (int): Size of each input sample
            (number of input channels).
        dim_out (int, optional): Size of each output sample
            (number of output channels).
            For OrthoConv (i.e. `use_lie=False`), this must be even.
            If not specified, defaults to `dim_in`.
        bias (bool, optional): If False, the layer will not
            learn an additive bias.
            (default: True)
        T (int, optional): Number of Taylor series terms used
            to approximate the matrix
            exponential. (default: 12)
        use_lie (bool, optional): If True,
            implements Lie OrthoConv; otherwise, it implements
            OrthoConv. (default: False)
        **kwargs: Additional arguments passed to the
            underlying MessagePassing layer.

    Shapes:
        - **Input:** node features of shape
          :math:`(|\mathcal{V}|, F_{\text{in}})` and edge
          indices of shape :math:`(2, |\mathcal{E}|)`
          or a sparse matrix of shape
          :math:`(|\mathcal{V}|, |\mathcal{V}|)`.
        - **Output:** node features of shape
          :math:`(|\mathcal{V}|, F_{\text{out}})`.
    """
    def __init__(self, dim_in, dim_out=None, bias=True, T=12, use_lie=False,
                 **kwargs):
        super().__init__()
        self.dim_in = dim_in
        if dim_out is None:
            self.dim_out = dim_in
            dim_out = dim_in
        else:
            self.dim_out = dim_out
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(dim_out))
        else:
            self.register_parameter('bias', None)

        if use_lie:
            base_conv = HermitianGCNConv
        else:
            base_conv = ComplexToRealGCNConv

        # For Lie OrthoConv, input and output dimensions must be equal.
        if use_lie and (dim_in != dim_out):
            raise ValueError(f"In Lie OrthoConv (use_lie=True), "
                             f"input_dim (dim_in={dim_in}) "
                             f"must equal output_dim (dim_out={dim_out}).")

        # For OrthoConv, output dimension must be even (features are paired).
        if not use_lie and (dim_out % 2 != 0):
            raise ValueError(f"For OrthoConv (use_lie=False), "
                             f"output dimension must be even. "
                             f"Got dim_out={dim_out}.")

        self.model = TaylorGCNConv(base_conv(dim_in, dim_out, **kwargs), T=T)

    def forward(self, x, edge_index):
        x = self.model(x, edge_index)
        if self.bias is not None:
            x = x + self.bias
        return x


class HermitianGCNConv(MessagePassing):
    _cached_edge_index: Optional[OptPairTensor]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: Optional[bool] = False,
        normalize: bool = True,
        bias: bool = False,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        if add_self_loops is None:
            add_self_loops = normalize

        if add_self_loops and not normalize:
            raise ValueError(f"'{self.__class__.__name__}' does not support "
                             f"adding self-loops to the graph when no "
                             f"on-the-fly normalization is applied")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        if in_channels != out_channels:
            raise ValueError(f"Input and output dimension must be the same"
                             f"for HermitianGCNConv. "
                             f"Got in_channels = {in_channels}, "
                             f"out_channels = {out_channels}")
        self.lin = torch.nn.Linear(in_channels, out_channels, bias=False)
        # below is used for counting the number of parameters
        self.lin.weight.upper_triangular_params = (in_channels *
                                                   (in_channels - 1) // 2)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin.weight.data = block_diagonal_init(self.lin.weight.data)
        self._cached_edge_index = None
        self._cached_adj_t = None
        if self.bias is not None:
            zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None, apply_feature_lin: bool = True,
                return_feature_only: bool = False) -> Tensor:

        if isinstance(x, (tuple, list)):
            raise ValueError(f"'{self.__class__.__name__}' received a tuple "
                             f"of node features as input while this layer "
                             f"does not support bipartite message passing. "
                             f"Please try other layers such as 'SAGEConv' or "
                             f"'GraphConv' instead")

        if apply_feature_lin:
            if return_feature_only:
                return x

        x = (x @ self.lin.weight - x @ self.lin.weight.T) / 2
        if self.bias is not None:
            x = x + self.bias

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow, x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow, x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)


class ComplexToRealGCNConv(MessagePassing):
    _cached_edge_index: Optional[OptPairTensor]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: Optional[bool] = False,
        normalize: bool = True,
        bias: bool = False,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        if add_self_loops is None:
            add_self_loops = normalize

        if add_self_loops and not normalize:
            raise ValueError(f"'{self.__class__.__name__}' does not support "
                             f"adding self-loops to the graph when no "
                             f"on-the-fly normalization is applied")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        if out_channels % 2 != 0:
            raise ValueError(f"Output dimension must be even "
                             f"for ComplexToRealGCNConv. Got "
                             f"out_channels = {out_channels}")
        self.lin = torch.nn.Linear(in_channels, out_channels, bias=False)

        multiplier = torch.ones(out_channels)
        multiplier[out_channels // 2:] = -1
        self.register_buffer('multiplier', multiplier.view(1, -1))

        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_channels // 2))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self._cached_edge_index = None
        self._cached_adj_t = None
        if self.bias is not None:
            zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None, apply_feature_lin: bool = True,
                return_feature_only: bool = False) -> Tensor:

        if isinstance(x, (tuple, list)):
            raise ValueError(f"'{self.__class__.__name__}' received a tuple "
                             f"of node features as input while this layer "
                             f"does not support bipartite message passing. "
                             f"Please try other layers such as 'SAGEConv' or "
                             f"'GraphConv' instead")

        if apply_feature_lin:
            x = self.lin(x)
            if self.bias is not None:
                x = x + self.bias
            if return_feature_only:
                return x

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow, x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow, x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        out = out * self.multiplier
        out = torch.roll(out, self.out_channels // 2, -1)

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)


class TaylorGCNConv(MessagePassing):
    def __init__(self, conv: HermitianGCNConv, T: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.conv = conv
        self.T = T

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None, **kwargs) -> Tensor:
        x = self.conv(x, edge_index, edge_weight, apply_feature_lin=True,
                      return_feature_only=True)
        x_k = x.clone()  # Create a copy of the input tensor

        for k in range(self.T):
            x_k = self.conv(x_k, edge_index, edge_weight,
                            apply_feature_lin=False, **kwargs) / (k + 1)
            x += x_k
        return x


@torch.jit._overload
def gcn_norm(  # noqa: F811
        edge_index, edge_weight, num_nodes, improved, add_self_loops, flow,
        dtype):
    # type: (Tensor, OptTensor, Optional[int], bool, bool, str, Optional[int]) -> OptPairTensor  # noqa
    pass


@torch.jit._overload
def gcn_norm(  # noqa: F811
        edge_index, edge_weight, num_nodes, improved, add_self_loops, flow,
        dtype):
    # type: (SparseTensor, OptTensor, Optional[int], bool, bool, str, Optional[int]) -> SparseTensor  # noqa
    pass


def gcn_norm(  # noqa: F811
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

        deg = torch_sparse.sum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = torch_sparse.mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = torch_sparse.mul(adj_t, deg_inv_sqrt.view(1, -1))

        return adj_t

    if is_torch_sparse_tensor(edge_index):
        assert edge_index.size(0) == edge_index.size(1)

        if edge_index.layout == torch.sparse_csc:
            raise NotImplementedError("Sparse CSC matrices are not yet "
                                      "supported in 'gcn_norm'")

        adj_t = edge_index
        if add_self_loops:
            adj_t, _ = add_self_loops_fn(adj_t, None, fill_value, num_nodes)

        edge_index, value = to_edge_index(adj_t)
        col, row = edge_index[0], edge_index[1]

        deg = scatter(value, col, 0, dim_size=num_nodes, reduce='sum')
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        value = deg_inv_sqrt[row] * value * deg_inv_sqrt[col]

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
    idx = col if flow == 'source_to_target' else row
    deg = scatter(edge_weight, idx, dim=0, dim_size=num_nodes, reduce='sum')
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    return edge_index, edge_weight
