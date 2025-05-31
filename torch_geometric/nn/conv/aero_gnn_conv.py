from typing import Optional

import torch
from torch import Tensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, ones
from torch_geometric.typing import Adj
from torch_geometric.utils import scatter


class AeroGNNConv(MessagePassing):
    r"""The AERO-GNN graph convolutional operator from the
    `"Towards Deep Attention in Graph Neural Networks:
    Problems and Remedies" <https://arxiv.org/abs/2306.02376>`_ paper.

    .. math::
        H^{(k)} =
        \begin{cases}
            \text{MLP}(X), & \text{if } k = 0, \\
            \mathcal{A}^{(k)} H^{(k-1)}, & \text{if } 1 \leq k \leq k_{max},
        \end{cases}

        Z^{(k)} = \sum_{\ell=0}^{k} \Gamma^{(\ell)} H^{(\ell)},
        \forall 1 \leq k \leq k_{max},

        Z^* = \sigma(Z^{(k_{max})}) W^*,

    where
    :math:`H^{(k)}` represents node features at layer :math:`k`,
    :math:`\mathcal{A}^{(k)}` is the edge attention matrix at layer :math:`k`,
    :math:`\Gamma^{(\ell)}` is the hop attention matrix,
    :math:`Z^*` is the final node representation,
    :math:`\sigma` is a non-linear activation function, and
    :math:`W^*` is an weight matrix.

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        out_channels (int): Size of each output sample.
        num_nodes (int): Number of nodes.
        num_heads (int): Number of attention heads.
        num_iters (int): Number of iterations.
        num_layers (int): Number of layers.
        dropout (float): Dropout rate.
        lambd (float): Decay rate.
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
            - node features :math:`(|\mathcal{V}|, F_{in})`,
            - edge indices :math:`(2, |\mathcal{E}|)`,
            - edge weights :math:`(|\mathcal{E}|)` *(optional)*
        - **output:**
            - node features :math:`(|\mathcal{V}|, F_{out})`
    """
    def __init__(
        self,
        in_channels: int,
        hid_channels: int,
        out_channels: int,
        num_nodes: int,
        num_heads: int,
        num_layers: int,
        num_iters: int,
        dropout: float,
        lambd: float,
        **kwargs,
    ):
        kwargs.setdefault("node_dim", 0)
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.hid_channels = hid_channels
        self.num_layers = num_layers
        self.num_nodes = num_nodes
        self.hid_channels_ = self.num_heads * self.hid_channels
        self.num_iters = num_iters
        self.dropout = dropout
        self.lambd = lambd
        self.setup_layers()
        self.reset_parameters()

    def setup_layers(self):
        self.dropout_layer = torch.nn.Dropout(self.dropout)
        self.elu = torch.nn.ELU()
        self.softplus = torch.nn.Softplus()

        self.dense_lins = torch.nn.ModuleList()
        self.atts = torch.nn.ParameterList()
        self.hop_atts = torch.nn.ParameterList()
        self.hop_biases = torch.nn.ParameterList()
        self.decay_weights = []

        self.dense_lins.append(
            Linear(
                self.in_channels,
                self.hid_channels_,
                bias=True,
                weight_initializer="glorot",
            ))
        for _ in range(self.num_layers - 1):
            self.dense_lins.append(
                Linear(
                    self.hid_channels_,
                    self.hid_channels_,
                    bias=True,
                    weight_initializer="glorot",
                ))
        self.dense_lins.append(
            Linear(
                self.hid_channels_,
                self.out_channels,
                bias=True,
                weight_initializer="glorot",
            ))

        for k in range(self.num_iters + 1):
            if k > 0:
                self.atts.append(
                    torch.nn.Parameter(
                        torch.Tensor(1, self.num_heads, self.hid_channels)))
                self.hop_atts.append(
                    torch.nn.Parameter(
                        torch.Tensor(1, self.num_heads,
                                     self.hid_channels * 2)))
            else:
                self.hop_atts.append(
                    torch.nn.Parameter(
                        torch.Tensor(1, self.num_heads, self.hid_channels)))
            self.hop_biases.append(
                torch.nn.Parameter(torch.Tensor(1, self.num_heads)))
            self.decay_weights.append(
                torch.log((self.lambd / (k + 1)) + (1 + 1e-6)))

    def reset_parameters(self):
        for lin in self.dense_lins:
            lin.reset_parameters()
        for att in self.atts:
            glorot(att)
        for att in self.hop_atts:
            glorot(att)
        for bias in self.hop_biases:
            ones(bias)

    def hid_feat_init(self, x: Tensor) -> Tensor:
        x = self.dropout_layer(x)
        x = self.dense_lins[0](x)

        for i_layer in range(self.num_layers - 1):
            x = self.elu(x)
            x = self.dropout_layer(x)
            x = self.dense_lins[i_layer + 1](x)

        return x

    def aero_propagate(self, h: Tensor, edge_index: Adj) -> Tensor:
        self.k = 0
        h = h.view(-1, self.num_heads, self.hid_channels)
        g = self.hop_att_pred(h, z_scale=None)
        z = h * g
        z_scale = z * self.decay_weights[self.k]

        for k in range(self.num_iters):
            self.k = k + 1
            h = self.propagate(edge_index, x=h, z_scale=z_scale)
            g = self.hop_att_pred(h, z_scale)
            z += h * g
            z_scale = z * self.decay_weights[self.k]

        return z

    def final_layer(self, z: Tensor) -> Tensor:
        z = z.view(-1, self.num_heads * self.hid_channels)
        z = self.elu(z)
        if self.dropout > 0:
            z = self.dropout_layer(z)
        z = self.dense_lins[-1](z)

        return z

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        h0 = self.hid_feat_init(x)
        z_k_max = self.aero_propagate(h0, edge_index)
        z_star = self.final_layer(z_k_max)

        return z_star

    def hop_att_pred(self, h: Tensor,
                     z_scale: Optional[Tensor] = None) -> Tensor:

        if z_scale is None:
            x = h
        else:
            x = torch.cat((h, z_scale), dim=-1)

        g = x.view(-1, self.num_heads, int(x.shape[-1]))
        g = self.elu(g)
        g = (self.hop_atts[self.k] * g).sum(dim=-1) + self.hop_biases[self.k]

        return g.unsqueeze(-1)

    def edge_att_pred(self, z_scale_i: Tensor, z_scale_j: Tensor,
                      edge_index: Adj) -> Tensor:

        # edge attention (alpha_check_ij)
        a_ij = z_scale_i + z_scale_j
        a_ij = self.elu(a_ij)
        a_ij = (self.atts[self.k - 1] * a_ij).sum(dim=-1)
        a_ij = self.softplus(a_ij) + 1e-6

        # symmetric normalization (alpha_ij)
        row, col = edge_index[0], edge_index[1]
        deg = scatter(a_ij, col, dim=0, dim_size=self.num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
        a_ij = deg_inv_sqrt[row] * a_ij * deg_inv_sqrt[col]

        return a_ij

    def message(self, edge_index: Adj, x_j: Tensor, z_scale_i: Tensor,
                z_scale_j: Tensor) -> Tensor:
        a = self.edge_att_pred(z_scale_i, z_scale_j, edge_index)
        return a.unsqueeze(-1) * x_j

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.in_channels}, "
            f"{self.out_channels}, hid_channels={self.hid_channels}, "
            f"num_heads={self.num_heads}, num_iterations={self.num_iters}, "
            f"dropout={self.dropout}, lambd={self.lambd})")
