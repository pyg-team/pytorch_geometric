from typing import Iterable, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Parameter, Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import add_remaining_self_loops
from torch_scatter import scatter
from torch_sparse import SparseTensor, matmul, fill_diag

from ..inits import glorot, zeros


class EGConv(MessagePassing):
    r"""The Efficient Graph Convolution from the `"Adaptive Filters and
    Aggregator Fusion for Efficient Graph Convolutions"
    <https://arxiv.org/abs/2104.01481>`_ paper

    Its node-wise formulation is given by:

    .. math::
        \mathbf{x}_i^{\prime} = {\LARGE ||}_{h=1}^H \sum_{\oplus \in \mathcal{A}}
        \sum_{b = 1}^B w_{i, h, \oplus, b} \;
        \underset{j \in \mathcal{N}(i) \cup \{i\}}{\bigoplus}
        \mathbf{\Theta}_b \mathbf{x}_{j}

    with :math:`\mathbf{\Theta}_b` denoting a basis weight,
    :math:`\oplus` denoting an aggregator, and :math:`w` denoting per-vertex
    weighting coefficients across different heads, bases, and aggregators.

    EGC retains O(V) memory usage, making it a sensible alternative to GCN,
    GraphSAGE, or GIN.

    .. note::
        For an example of using :obj:`EGConv`, see `examples/egc.py
        <https://github.com/rusty1s/pytorch_geometric/blob/master/
        examples/egc.py>`_.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        aggrs (Iterable[str], optional): Aggregators to be used. Supported
            aggregators are :obj:`["sum", "mean", "symnorm", "max", "min",
            "std", "var"]. Defaults to the EGC-S layer in the paper.
            Multiple aggregators can be used to improve performance.
            (default: :obj:`("symnorm",)`)
        num_heads (int, optional): Number of heads to use. Must have
            :obj:`out_channels % num_heads == 0`. Recommended to set
            num_heads >= num_bases. (default: :obj:`8`)
        num_bases (int, optional): Number of basis weights to use.
            (default: :obj:`4`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of the edge index with added self loops on first
            execution, along with caching the calculation of the symmetric
            normalized edge weights if the :obj:`"symnorm"` aggregator is
            being used. This parameter should only be set to :obj:`True` in
            transductive learning scenarios. (default: :obj:`False`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    _cached_edge_index: Optional[Tuple[Tensor, OptTensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aggrs: Iterable[str] = ("symnorm", ),
        num_heads: int = 8,
        num_bases: int = 4,
        cached: bool = False,
        add_self_loops: bool = True,
        bias: bool = True,
        **kwargs,
    ):
        super(EGConv, self).__init__(node_dim=0, **kwargs)

        if out_channels % num_heads != 0:
            raise ValueError(
                "out_channels must be divisible by the number of heads")

        for a in aggrs:
            if a not in {"sum", "mean", "symnorm", "min", "max", "var", "std"}:
                raise ValueError("Unsupported aggregator: {}".format(a))

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.num_bases = num_bases
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.aggregators = list(aggrs)

        self.bases_weight = Parameter(
            torch.Tensor(in_channels, (out_channels // num_heads) * num_bases))
        self.comb_weight = Linear(in_channels,
                                  num_heads * num_bases * len(aggrs))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.bases_weight)
        self.comb_weight.reset_parameters()
        zeros(self.bias)
        self._cached_adj_t = None
        self._cached_edge_index = None

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        """"""
        symnorm_weight: OptTensor = None
        if "symnorm" in self.aggregators:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, symnorm_weight = gcn_norm(  # yapf: disable
                        edge_index, None, num_nodes=x.size(self.node_dim),
                        improved=False, add_self_loops=self.add_self_loops
                    )
                    if self.cached:
                        self._cached_edge_index = (edge_index, symnorm_weight)
                else:
                    edge_index, symnorm_weight = cache

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, None, num_nodes=x.size(self.node_dim),
                        improved=False, add_self_loops=self.add_self_loops
                    )
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        elif self.add_self_loops:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if self.cached and cache is not None:
                    edge_index = cache[0]
                else:
                    edge_index, _ = add_remaining_self_loops(edge_index)
                    if self.cached:
                        self._cached_edge_index = (edge_index, None)

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if self.cached and cache is not None:
                    edge_index = cache
                else:
                    edge_index = fill_diag(edge_index, 1.0)
                    if self.cached:
                        self._cached_adj_t = edge_index

        num_nodes = x.size(0)

        # [num_nodes, (out_channels // num_heads) * num_bases]
        bases = torch.matmul(x, self.bases_weight)
        # [num_nodes, num_heads * num_bases * num_aggrs]
        weightings = self.comb_weight(x)

        if symnorm_weight is not None:
            symnorm_weight = symnorm_weight.view(-1, 1)

        # [num_nodes, num_aggregators, (out_channels // num_heads) * num_bases]
        # propagate_type: (x: Tensor, symnorm_weight: OptTensor)
        aggregated = self.propagate(edge_index, x=bases,
                                    symnorm_weight=symnorm_weight, size=None)

        weightings = weightings.view(num_nodes, self.num_heads,
                                     self.num_bases * len(self.aggregators))
        aggregated = aggregated.view(
            num_nodes,
            len(self.aggregators) * self.num_bases,
            self.out_channels // self.num_heads,
        )

        # [num_nodes, num_heads, out_channels // num_heads]
        out = torch.matmul(weightings, aggregated)
        out = out.view(num_nodes, self.out_channels)
        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def aggregate(
        self,
        inputs: Tensor,
        index: Tensor,
        dim_size: Optional[int] = None,
        symnorm_weight: OptTensor = None,
    ) -> Tensor:
        aggregated = []
        for aggregator in self.aggregators:
            if aggregator == "sum":
                out = scatter(inputs, index, 0, None, dim_size, reduce="sum")
            elif aggregator == "symnorm":
                assert symnorm_weight is not None
                out = scatter(inputs * symnorm_weight, index, 0, None,
                              dim_size, reduce="sum")
            elif aggregator == "mean":
                out = scatter(inputs, index, 0, None, dim_size, reduce="mean")
            elif aggregator == "min":
                out = scatter(inputs, index, 0, None, dim_size, reduce="min")
            elif aggregator == "max":
                out = scatter(inputs, index, 0, None, dim_size, reduce="max")
            elif aggregator == "var" or aggregator == "std":
                mean = scatter(inputs, index, 0, None, dim_size, reduce="mean")
                mean_squares = scatter(inputs * inputs, index, 0, None,
                                       dim_size, reduce="mean")
                out = mean_squares - mean * mean
                if aggregator == "std":
                    out = torch.sqrt(torch.relu(out) + 1e-5)
            else:
                raise ValueError(f'Unknown aggregator "{aggregator}".')
            aggregated.append(out)

        return torch.stack(aggregated, dim=1)

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        aggregated = []
        if len(self.aggregators) > 1 and "symnorm" in self.aggregators:
            adj_t_nonorm = adj_t.set_value(None)
        else:
            # No normalization is calculated in forward if symnorm isn't one
            # of the aggregators
            adj_t_nonorm = adj_t

        for aggregator in self.aggregators:
            if aggregator == "symnorm":
                correct_adj = adj_t
                agg = "sum"
            else:
                correct_adj = adj_t_nonorm
                agg = aggregator

            if aggregator in ["var", "std"]:
                mean = matmul(correct_adj, x, reduce="mean")
                mean_sq = matmul(correct_adj, x * x, reduce="mean")
                out = mean_sq - mean * mean
                if aggregator == "std":
                    out = torch.sqrt(torch.relu(out) + 1e-5)
                aggregated.append(out)
            else:
                aggregated.append(matmul(correct_adj, x, reduce=agg))

        return torch.stack(aggregated, dim=1)

    def __repr__(self):
        return "{}({}, {}, {})".format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels,
            self.aggregators,
        )
