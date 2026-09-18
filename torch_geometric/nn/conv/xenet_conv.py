from collections.abc import Sequence
from typing import Union

import torch
from torch import Tensor, nn

from torch_geometric.nn.conv.message_passing import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import reset


class XENetConv(MessagePassing):
    r"""XENet graph convolution.

    XENet uses node features and edge features to compute edge-level
    representations, which are then aggregated in incoming and outgoing
    directions to update node features.

    Args:
        stack_channels (int or list[int]): Hidden channels of the
            edge-message stack.
        node_channels (int): Number of output node channels.
        edge_channels (int): Number of output edge channels.
        attention (bool, optional): Whether to use attention during
            message aggregation. (default: ``True``)
        bias (bool, optional): Whether to use bias terms.
            (default: ``True``)
    """
    def __init__(
        self,
        stack_channels: Union[int, Sequence[int]],
        node_channels: int,
        edge_channels: int,
        attention: bool = True,
        bias: bool = True,
    ):
        super().__init__(aggr="sum")

        if isinstance(stack_channels, int):
            stack_channels = [stack_channels]

        if len(stack_channels) == 0:
            raise ValueError(
                "'stack_channels' must contain at least one layer")

        self.stack_channels = list(stack_channels)
        self.node_channels = node_channels
        self.edge_channels = edge_channels
        self.attention = attention

        self.stack_models = nn.ModuleList()
        self.stack_activations = nn.ModuleList()

        for i, channels in enumerate(self.stack_channels):
            in_channels = -1 if i == 0 else self.stack_channels[i - 1]

            self.stack_models.append(Linear(in_channels, channels, bias=bias))
            self.stack_activations.append(nn.PReLU(channels))

        self.node_model = Linear(
            -1,
            node_channels,
            bias=bias,
        )

        self.node_activation = nn.PReLU(node_channels)

        self.edge_model = Linear(
            self.stack_channels[-1],
            edge_channels,
            bias=bias,
        )

        self.edge_activation = nn.PReLU(edge_channels)

        if attention:
            self.incoming_attention = Linear(
                self.stack_channels[-1],
                1,
                bias=bias,
            )
            self.outgoing_attention = Linear(
                self.stack_channels[-1],
                1,
                bias=bias,
            )

    def reset_parameters(self):
        super().reset_parameters()

        for model in self.stack_models:
            reset(model)

        for activation in self.stack_activations:
            reset(activation)
        reset(self.node_model)
        reset(self.node_activation)
        reset(self.edge_model)
        reset(self.edge_activation)

        if self.attention:
            reset(self.incoming_attention)
            reset(self.outgoing_attention)

    def _compute_stack(
        self,
        x_i: Tensor,
        x_j: Tensor,
        e_ij: Tensor,
        e_ji: Tensor,
    ) -> Tensor:
        """Computes the XENet edge representation s_ij."""
        stack = torch.cat(
            [x_i, x_j, e_ij, e_ji],
            dim=-1,
        )

        for model, activation in zip(
                self.stack_models,
                self.stack_activations,
        ):
            stack = model(stack)
            stack = activation(stack)

        return stack

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Runs the forward pass."""
        if x.dim() != 2:
            raise ValueError(
                "'x' must have shape [num_nodes, num_node_features]")

        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError("'edge_index' must have shape [2, num_edges]")

        if edge_attr.dim() == 1:
            edge_attr = edge_attr.view(-1, 1)

        if edge_attr.dim() != 2:
            raise ValueError(
                "'edge_attr' must have shape [num_edges, num_edge_features]")

        if edge_index.size(1) != edge_attr.size(0):
            raise ValueError(
                "'edge_index' and 'edge_attr' must contain the same "
                "number of edges")

        reverse_index = self._get_reverse_edge_index(
            edge_index,
            x.size(0),
        )

        src, dst = edge_index

        x_i = x[src]
        x_j = x[dst]

        e_ij = edge_attr
        e_ji = edge_attr[reverse_index]

        stack = self._compute_stack(
            x_i,
            x_j,
            e_ij,
            e_ji,
        )

        if self.attention:
            outgoing_alpha = torch.sigmoid(self.outgoing_attention(stack))
            incoming_alpha = torch.sigmoid(
                self.incoming_attention(stack[reverse_index]))
        else:
            outgoing_alpha = stack.new_ones(stack.size(0), 1)
            incoming_alpha = stack.new_ones(stack.size(0), 1)

        outgoing = self.propagate(
            edge_index.flip(0),
            stack=stack,
            alpha=outgoing_alpha,
            size=(x.size(0), x.size(0)),
        )

        incoming = self.propagate(
            edge_index,
            stack=stack,
            alpha=incoming_alpha,
            size=(x.size(0), x.size(0)),
        )

        # Node update:
        # x'_i = phi_n(x_i || s_i_out || s_i_in)
        x_out = self.node_model(torch.cat(
            [x, outgoing, incoming],
            dim=-1,
        ))
        x_out = self.node_activation(x_out)

        # Edge update:
        # e'_ij = phi_e(s_ij)
        edge_out = self.edge_model(stack)
        edge_out = self.edge_activation(edge_out)

        return x_out, edge_out

    def message(self, stack: Tensor, alpha: Tensor) -> Tensor:
        return stack * alpha

    @staticmethod
    def _get_reverse_edge_index(
        edge_index: Tensor,
        num_nodes: int,
    ) -> Tensor:
        """Returns the index of the reverse edge for every edge."""
        src, dst = edge_index

        # Encode each directed edge (src, dst) as a unique integer.
        keys = src * num_nodes + dst
        reverse_keys = dst * num_nodes + src

        # Sort edge keys so we can efficiently locate reverse edges.
        sorted_keys, permutation = torch.sort(keys)

        positions = torch.searchsorted(
            sorted_keys,
            reverse_keys,
        )

        valid = positions < sorted_keys.numel()

        if valid.any():
            valid_positions = positions[valid]
            matches = (sorted_keys[valid_positions] == reverse_keys[valid])
            valid_indices = valid.nonzero(as_tuple=True)[0]
            valid[valid_indices] = matches

        if not valid.all():
            raise ValueError(
                "XENetConv requires both directions of every edge. "
                "For every edge i -> j, the reverse edge j -> i "
                "must also be present in edge_index.")

        return permutation[positions]

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"stack_channels={self.stack_channels}, "
                f"node_channels={self.node_channels}, "
                f"edge_channels={self.edge_channels})")
