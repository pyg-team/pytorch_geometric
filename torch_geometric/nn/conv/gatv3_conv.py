import math
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import Adj, OptTensor, PairTensor, SparseTensor
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax


class GATv3Conv(MessagePassing):
    """GATv3Conv implements a context-aware attention mechanism
    inspired by the GATher framework [1], extending GATv2 with
    element-wise feature multiplication,
    optional weight sharing, and scaling.

    Reference:
        [1] "GATher: Graph Attention Based Predictions of Gene-Disease Links,"
            Narganes-Carlon et al., 2024 (arXiv:2409.16327).
    """

    _alpha: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        dropout: float = 0.0,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = "mean",
        name: str = "unnamed",
        *,
        concat: bool = True,
        bias: bool = True,
        share_weights: bool = False,
        **kwargs,
    ):
        super().__init__(node_dim=0, aggr="sum", flow="source_to_target",
                         **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        self.fill_value = fill_value
        self.share_weights = share_weights
        self.sigmoid = torch.nn.Sigmoid()
        self.attention_sigmoid = torch.nn.Sigmoid()
        self.name = name

        # Define my layer that will scale the attentions
        self.context_attention_layer = torch.nn.Linear(
            self.out_channels,
            1,
            bias=True,
        )

        if isinstance(in_channels, int):
            self.lin_l = Linear(
                in_channels,
                heads * out_channels,
                bias=bias,
                weight_initializer="glorot",
            )
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = Linear(
                    in_channels,
                    heads * out_channels,
                    bias=bias,
                    weight_initializer="glorot",
                )
        else:
            self.lin_l = Linear(
                in_channels[0],
                heads * out_channels,
                bias=bias,
                weight_initializer="glorot",
            )
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = Linear(
                    in_channels[1],
                    heads * out_channels,
                    bias=bias,
                    weight_initializer="glorot",
                )

        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False,
                                   weight_initializer="glorot")
        else:
            self.lin_edge = None

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        if bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self._alpha = None
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        zeros(self.bias)

    def forward(
        self,
        x: Union[Tensor, PairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        *,
        return_attention_weights: bool = None,
    ):

        x_l: OptTensor = None
        x_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2
            x_l = self.lin_l(x).view(-1, self.heads, self.out_channels)
            if self.share_weights:
                x_r = x_l
            else:
                x_r = self.lin_r(x).view(-1, self.heads, self.out_channels)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2
            x_l = self.lin_l(x_l).view(-1, self.heads, self.out_channels)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, self.heads, self.out_channels)

        # Remove and add self loops
        # The original code did not check if the graph is bipartite
        if isinstance(x, tuple):
            pass
        else:
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        out = self.propagate(
            edge_index,
            x=(x_l, x_r),
            edge_attr=edge_attr,
            size=(x_l.shape[0], x_r.shape[0]),
        )

        alpha = self._alpha
        self._alpha = None

        # Take the mean of the output representation for each of the heads
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            if isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout="coo")
        return out

    def message(self, x_j: Tensor, x_i: Tensor, edge_index: Tensor) -> Tensor:

        # Self dot product from Vaswani et al. Dec 2017
        # https://arxiv.org/pdf/1706.03762.pdf
        alpha = x_i * x_j
        alpha = self.context_attention_layer(alpha).squeeze(-1)

        # Equation 1 from Vaswani et al. Dec 2017
        # https://arxiv.org/pdf/1706.03762.pdf
        norm = math.sqrt(x_i.size(-1))
        alpha = alpha / norm
        alpha = softmax(alpha, index=edge_index[1, :], ptr=None)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        alpha = alpha.unsqueeze(-1)  # Fix shapes

        return x_j * alpha

    def aggregate(
        self,
        inputs: Tensor,
        index: Tensor,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
    ) -> Tensor:
        return super().aggregate(inputs, index, ptr, dim_size)

    def update(self, inputs: Tensor) -> Tensor:
        return super().update(inputs)

    def __repr__(self) -> str:
        return f"GATv3Conv({self.in_channels}, {self.out_channels}, heads={self.heads})"
