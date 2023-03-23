from typing import Dict, Literal, Optional

import torch
from torch import Tensor

from torch_geometric.contrib.nn.conv import (
    SE3Conv,
    SE3GATConv,
    SE3Norm,
)
from torch_geometric.contrib.utils import Fiber
from torch_geometric.contrib.utils import get_basis
from torch_geometric.data import Data
from torch_geometric.nn.aggr import MaxAggregation, MeanAggregation
from torch_geometric.typing import Adj


class SE3Transformer(torch.nn.Module):
    r"""The SE(3)-Equivariant Graph Transformer from the
    `"SE(3)-Transformers: 3D Roto-Translation Equivariant Attention Networks"
    <https://arxiv.org/abs/2006.10503>`_ paper. Is composed of
    :class:`~torch_geometric.contrib.nn.conv.SE3Conv` and
    :class:`~torch_geometric.contrib.nn.conv.SE3GATConv` layers, with optional
    pooling and MLP layers at the end.

    .. note::

        For an example of how to use this model, see
        `examples/contrib/se3transformer_qm9.py`_.

    Args:
        num_layers (int): Number of attention layers
        fiber_in (:class:`torch_geometric.contrib.utils.Fiber`): Input fiber description
        fiber_hidden (:class:`torch_geometric.contrib.utils.Fiber`): Hidden fiber description
        fiber_out (:class:`torch_geometric.contrib.utils.Fiber`): Output fiber description
        num_heads (int): Number of attention heads
        channels_div (int): Channels division before feeding to the attention layer
        fiber_edge (:class:`torch_geometric.contrib.utils.Fiber`): Input edge fiber description
        return_type (int, optional): Return only features of this type
        pooling (str, optional): Graph pooling before MLP layers. Valid options
            are 'avg' and 'max'
        output_dim (int): Output dim if pooling
        use_layer_norm (bool): Apply layer normalization between MLP layers
        norm (bool): Apply a normalization between MLP layers
    """

    def __init__(
        self,
        num_layers: int,
        fiber_in: Fiber,
        fiber_hidden: Fiber,
        fiber_out: Fiber,
        num_heads: int,
        channels_div: int,
        fiber_edge: Fiber = Fiber({}),
        return_type: Optional[int] = None,
        pooling: Optional[Literal["avg", "max"]] = None,
        output_dim: Optional[int] = None,
        use_layer_norm: bool = True,
        norm: bool = True,
    ):
        super().__init__()
        self.pooling = pooling
        self.return_type = return_type
        self.max_degree = max(*fiber_in.degrees, *fiber_hidden.degrees, *fiber_out.degrees)


        graph_modules = []
        for i in range(num_layers):
            graph_modules.append(
                SE3GATConv(
                    fiber_in=fiber_in,
                    fiber_out=fiber_hidden,
                    fiber_edge=fiber_edge,
                    num_heads=num_heads,
                    channels_div=channels_div,
                    use_layer_norm=use_layer_norm,
                )
            )
            if norm:
                graph_modules.append(SE3Norm(fiber_hidden))
            fiber_in = fiber_hidden


        graph_modules.append(
            SE3Conv(
                fiber_in=fiber_in,
                fiber_out=fiber_out,
                fiber_edge=fiber_edge,
                use_layer_norm=use_layer_norm,
                self_interaction=True,
            )
        )
        self.graph_modules = Sequential(
            *graph_modules
        )  # Use plain module list instead?
        if pooling is not None:
            assert return_type is not None, "return_type must be specified when pooling"
            self.pooling_module = self._get_pooling_module()
            n_out_features = fiber_out.num_features
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(n_out_features, n_out_features),
                torch.nn.ReLU(),
                torch.nn.Linear(n_out_features, output_dim),
            )

    def forward(self, data: Data):

        # Compute bases in case they weren't precomputed as part of the data loading
        if "basis" not in data:
            data.basis = get_basis(relative_pos=data.edge_attr[:, -3:])

        transformed_data = self.graph_modules(
            data.node_feats, data.edge_feats, data.edge_index, basis=data.basis
        )

        if self.pooling is not None:
            out = self.pooling_module(
                transformed_data[str(self.return_type)], index=data.batch, dim=0
            )
            out = self.mlp(out.squeeze(-1)).squeeze(-1)
            return out

        if self.return_type is not None:
            return transformed_data[str(self.return_type)]

        return transformed_data

    def _get_pooling_module(self):
        assert self.pooling in ["max", "avg"], f"Unknown pooling: {self.pool}"
        assert (
            self.return_type == 0 or self.pooling == "avg"
        ), "Max pooling on type > 0 features will break equivariance"
        return MaxAggregation() if self.pooling == "max" else MeanAggregation()

class Sequential(torch.nn.Sequential):
    """Sequential module with arbitrary forward args and kwargs. Used to pass graph, basis, and edge features."""

    def forward(self, input, *args, **kwargs):
        for module in self:
            input = module(input, *args, **kwargs)
        return input
