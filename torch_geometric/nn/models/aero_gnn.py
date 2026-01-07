from typing import Optional

import torch
from torch import Tensor

from torch_geometric.nn.conv import AEROConv
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj


class AEROGNN(torch.nn.Module):
    r"""The AERO-GNN (Attentive dEep pROpagation-GNN) model from the
    `"Towards Deep Attention in Graph Neural Networks: Problems and Remedies"
    <https://arxiv.org/abs/2306.02376>`_ paper.

    AERO-GNN addresses problems in deep graph attention, including vulnerability
    to over-smoothed features and smooth cumulative attention. The model mitigates
    these issues through:

    1. **Edge attention** :math:`\alpha_{ij}^{(k)}`: Learns adaptive attention
       weights for edges at each propagation iteration, remaining edge-adaptive
       and graph-adaptive even at deep layers.
    2. **Hop attention** :math:`\gamma_i^{(k)}`: Learns adaptive attention weights
       for each propagation hop, remaining hop-adaptive, node-adaptive, and
       graph-adaptive at deep layers.
    3. **Exponential decay weights**: Applied across propagation iterations to
       prevent attention collapse and maintain expressiveness at deep layers.

    The model first applies multi-layer MLP transformations to input features,
    then performs iterative message passing with AERO attention mechanisms,
    and finally applies a classification layer.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of MLP layers for feature transformation.
            (default: :obj:`1`)
        out_channels (int, optional): Size of each output sample. If set to
            :obj:`None`, will be set to :obj:`hidden_channels`.
            (default: :obj:`None`)
        iterations (int, optional): Number of propagation iterations :math:`K`.
            (default: :obj:`10`)
        heads (int, optional): Number of multi-head attentions. (default: :obj:`1`)
        lambd (float, optional): Decay weight parameter :math:`\lambda` for
            exponential decay. (default: :obj:`1.0`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.0`)
        add_dropout (bool, optional): If set to :obj:`True`, applies dropout
            before the final classification layer. (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})`,
          edge indices :math:`(2, |\mathcal{E}|)`
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})`
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int = 1,
        out_channels: Optional[int] = None,
        iterations: int = 10,
        heads: int = 1,
        lambd: float = 1.0,
        dropout: float = 0.0,
        add_dropout: bool = False,
        bias: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.out_channels = out_channels or hidden_channels
        self.iterations = iterations
        self.heads = heads
        self.lambd = lambd
        self.dropout = dropout
        self.add_dropout = add_dropout

        # Compute hidden channels with heads
        self.hidden_channels_with_heads = heads * hidden_channels

        # MLP layers for feature transformation
        self.lins = torch.nn.ModuleList()
        # First layer: in_channels -> hidden_channels * heads
        self.lins.append(
            Linear(
                in_channels,
                self.hidden_channels_with_heads,
                bias=bias,
                weight_initializer='glorot',
            ))
        # Middle layers: hidden_channels * heads -> hidden_channels * heads
        for _ in range(num_layers - 1):
            self.lins.append(
                Linear(
                    self.hidden_channels_with_heads,
                    self.hidden_channels_with_heads,
                    bias=bias,
                    weight_initializer='glorot',
                ))
        # Final layer: hidden_channels * heads -> out_channels
        self.lins.append(
            Linear(
                self.hidden_channels_with_heads,
                self.out_channels,
                bias=bias,
                weight_initializer='glorot',
            ))

        # AERO convolution layer for iterative propagation
        self.conv = AEROConv(
            in_channels=self.hidden_channels_with_heads,
            out_channels=hidden_channels,
            heads=heads,
            iterations=iterations,
            lambd=lambd,
            dropout=dropout,
            bias=bias,
        )

        self.dropout_layer = torch.nn.Dropout(p=dropout)
        self.elu = torch.nn.ELU()

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for lin in self.lins:
            lin.reset_parameters()
        self.conv.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        num_nodes: Optional[int] = None,
    ) -> Tensor:
        r"""Runs the forward pass of the module.

        Args:
            x (torch.Tensor): The input node features of shape
                :math:`(|\mathcal{V}|, F_{in})`.
            edge_index (torch.Tensor or SparseTensor): The edge indices.
            num_nodes (int, optional): The number of nodes. If not provided,
                will be inferred from :obj:`edge_index`. (default: :obj:`None`)

        Returns:
            torch.Tensor: The output node features of shape
                :math:`(|\mathcal{V}|, F_{out})`.
        """
        # Feature transformation: MLP layers
        x = self.dropout_layer(x)
        x = self.lins[0](x)

        # Apply ELU and dropout for middle layers
        for i in range(1, self.num_layers):
            x = self.elu(x)
            x = self.dropout_layer(x)
            x = self.lins[i](x)

        # AERO propagation: iterative message passing with attention
        x = self.conv(x, edge_index, num_nodes=num_nodes)

        # Final classification layer
        x = x.view(-1, self.hidden_channels_with_heads)
        x = self.elu(x)
        if self.add_dropout:
            x = self.dropout_layer(x)
        x = self.lins[-1](x)

        return x

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}({self.in_channels}, '
            f'{self.hidden_channels}, num_layers={self.num_layers}, '
            f'out_channels={self.out_channels}, iterations={self.iterations}, '
            f'heads={self.heads})')
