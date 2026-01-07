import math
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, ones
from torch_geometric.typing import Adj, OptTensor, SparseTensor
from torch_geometric.utils import scatter


class AEROConv(MessagePassing):
    r"""The AERO (Attentive dEep pROpagation) graph convolution operator from
    the `"Towards Deep Attention in Graph Neural Networks: Problems and Remedies"
    <https://arxiv.org/abs/2306.02376>`_ paper.

    AERO-GNN addresses problems in deep graph attention, including vulnerability
    to over-smoothed features and smooth cumulative attention. The AERO operator
    performs iterative message passing with attention mechanisms that include:

    1. **Edge attention** :math:`\alpha_{ij}^{(k)}`: Computes attention weights
       for edges at iteration :math:`k` using:
       .. math::
           \alpha_{ij}^{(k)} = \text{softplus}(\mathbf{a}^{(k)} \cdot
           (\mathbf{z}_i^{(k)} + \mathbf{z}_j^{(k)})) + \epsilon

       followed by symmetric normalization:
       .. math::
           \hat{\alpha}_{ij}^{(k)} = \frac{\alpha_{ij}^{(k)}}{\sqrt{\deg(i) \deg(j)}}

    2. **Hop attention** :math:`\gamma_i^{(k)}`: Computes attention weights for
       each propagation hop:
       .. math::
           \gamma_i^{(k)} = \text{ELU}(\mathbf{W}^{(k)} [\mathbf{h}_i^{(k)},
           \mathbf{z}_i^{(k)}]) + \mathbf{b}^{(k)}

    3. **Decay weights**: Applies exponential decay across propagation iterations:
       .. math::
           w_k = \log\left(\frac{\lambda}{k+1} + 1 + \epsilon\right)

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head attentions. (default: :obj:`1`)
        iterations (int, optional): Number of propagation iterations :math:`K`.
            (default: :obj:`10`)
        lambd (float, optional): Decay weight parameter :math:`\lambda` for
            exponential decay. (default: :obj:`1.0`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients. (default: :obj:`0.0`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})`,
          edge indices :math:`(2, |\mathcal{E}|)`
        - **output:** node features :math:`(|\mathcal{V}|, H * F_{out})` where
          :math:`H` is the number of heads
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        iterations: int = 10,
        lambd: float = 1.0,
        dropout: float = 0.0,
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.iterations = iterations
        self.lambd = lambd
        self.dropout = dropout

        # Edge attention parameters: one for each iteration k in [1, K]
        self.edge_atts = torch.nn.ParameterList([
            Parameter(torch.empty(1, heads, out_channels))
            for _ in range(iterations)
        ])

        # Hop attention parameters: one for each iteration k in [0, K]
        # For k=0, we only use h (no z_scale), so dimension is out_channels
        # For k>0, we concatenate h and z_scale, so dimension is 2*out_channels
        self.hop_atts = torch.nn.ParameterList([
            Parameter(torch.empty(1, heads, out_channels))
        ])
        for _ in range(iterations):
            self.hop_atts.append(
                Parameter(torch.empty(1, heads, 2 * out_channels))
            )

        # Hop attention biases
        self.hop_biases = torch.nn.ParameterList([
            Parameter(torch.empty(1, heads))
            for _ in range(iterations + 1)
        ])

        # Decay weights: pre-computed log values for efficiency
        self.register_buffer(
            'decay_weights',
            torch.tensor([
                math.log((lambd / (k + 1)) + (1 + 1e-6))
                for k in range(iterations + 1)
            ], dtype=torch.float32)
        )

        if bias:
            self.bias = Parameter(torch.empty(heads * out_channels))
        else:
            self.register_parameter('bias', None)

        # Current iteration index (set during forward pass)
        self._current_iteration = 0
        # Store edge_index during forward pass for message function
        self._edge_index: Optional[Tensor] = None

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for att in self.edge_atts:
            glorot(att)
        for att in self.hop_atts:
            glorot(att)
        for bias in self.hop_biases:
            ones(bias)
        if self.bias is not None:
            ones(self.bias)

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        num_nodes: Optional[int] = None,
    ) -> Tensor:
        r"""Runs the forward pass of the module.

        Args:
            x (torch.Tensor): The input node features of shape
                :math:`(|\mathcal{V}|, H * F_{out})` where :math:`H` is the
                number of heads and :math:`F_{out}` is the output feature size.
            edge_index (torch.Tensor or SparseTensor): The edge indices.
            num_nodes (int, optional): The number of nodes. If not provided,
                will be inferred from :obj:`edge_index`. (default: :obj:`None`)

        Returns:
            torch.Tensor: The output node features of shape
                :math:`(|\mathcal{V}|, H * F_{out})`.
        """
        if num_nodes is None:
            if isinstance(edge_index, Tensor):
                num_nodes = int(edge_index.max()) + 1
            elif isinstance(edge_index, SparseTensor):
                num_nodes = edge_index.size(0)
            else:
                raise ValueError("Cannot infer num_nodes from edge_index type")

        # Reshape input to (num_nodes, heads, out_channels)
        h = x.view(-1, self.heads, self.out_channels)

        # Initialize: k=0
        self._current_iteration = 0
        g = self._hop_attention(h, z_scale=None)
        z = h * g
        z_scale = z * self.decay_weights[0].item()

        # Store edge_index for use in message function
        if isinstance(edge_index, Tensor):
            self._edge_index = edge_index
        else:
            # For SparseTensor, convert to edge_index
            row, col, _ = edge_index.coo()
            self._edge_index = torch.stack([row, col], dim=0)

        # Iterative propagation: k in [1, K]
        for k in range(1, self.iterations + 1):
            self._current_iteration = k
            # Propagate messages
            # MessagePassing will split z_scale into z_scale_i and z_scale_j
            h = self.propagate(
                edge_index,
                x=h,
                z_scale=z_scale,
                num_nodes=num_nodes,
                size=None,
            )
            # Compute hop attention and accumulate
            g = self._hop_attention(h, z_scale)
            z = z + h * g
            # Update z_scale for next iteration
            z_scale = z * self.decay_weights[k].item()

        # Reshape back to (num_nodes, heads * out_channels)
        out = z.view(-1, self.heads * self.out_channels)

        if self.bias is not None:
            out = out + self.bias

        # Clear stored edge_index
        self._edge_index = None

        return out

    def _hop_attention(
        self,
        h: Tensor,
        z_scale: Optional[Tensor],
    ) -> Tensor:
        r"""Computes hop attention weights.

        Args:
            h (torch.Tensor): Current node features of shape
                :math:`(|\mathcal{V}|, H, F_{out})`.
            z_scale (torch.Tensor, optional): Scaled accumulated features of
                shape :math:`(|\mathcal{V}|, H, F_{out})`. If :obj:`None`,
                only :obj:`h` is used (for k=0).

        Returns:
            torch.Tensor: Hop attention weights of shape
                :math:`(|\mathcal{V}|, H, 1)`.
        """
        k = self._current_iteration

        if z_scale is None:
            # k=0: only use h
            x = h
        else:
            # k>0: concatenate h and z_scale
            x = torch.cat([h, z_scale], dim=-1)

        # Apply ELU activation
        x = F.elu(x)

        # Compute attention: (hop_att * x).sum(dim=-1) + bias
        att = self.hop_atts[k]
        g = (att * x).sum(dim=-1) + self.hop_biases[k]

        return g.unsqueeze(-1)

    def _edge_attention(
        self,
        z_scale_i: Tensor,
        z_scale_j: Tensor,
        edge_index: Tensor,
        num_nodes: int,
    ) -> Tensor:
        r"""Computes edge attention weights with symmetric normalization.

        Args:
            z_scale_i (torch.Tensor): Scaled features for target nodes of shape
                :math:`(|\mathcal{E}|, H, F_{out})`.
            z_scale_j (torch.Tensor): Scaled features for source nodes of shape
                :math:`(|\mathcal{E}|, H, F_{out})`.
            edge_index (torch.Tensor): Edge indices of shape :math:`(2, |\mathcal{E}|)`.
            num_nodes (int): Number of nodes.

        Returns:
            torch.Tensor: Normalized edge attention weights of shape
                :math:`(|\mathcal{E}|,)`.
        """
        k = self._current_iteration

        # Compute unnormalized attention: a_ij = softplus(att^T (z_i + z_j)) + eps
        a_ij = z_scale_i + z_scale_j
        a_ij = F.elu(a_ij)
        a_ij = (self.edge_atts[k - 1] * a_ij).sum(dim=-1)
        a_ij = F.softplus(a_ij) + 1e-6

        # Symmetric normalization: a_ij / sqrt(deg(i) * deg(j))
        row, col = edge_index[0], edge_index[1]
        # Compute degrees for both source and target nodes
        deg_col = scatter(a_ij, col, dim=0, dim_size=num_nodes, reduce='sum')
        deg_row = scatter(a_ij, row, dim=0, dim_size=num_nodes, reduce='sum')
        
        deg_col_inv_sqrt = deg_col.pow(-0.5)
        deg_col_inv_sqrt = deg_col_inv_sqrt.masked_fill(
            deg_col_inv_sqrt == float('inf'), 0.0
        )
        deg_row_inv_sqrt = deg_row.pow(-0.5)
        deg_row_inv_sqrt = deg_row_inv_sqrt.masked_fill(
            deg_row_inv_sqrt == float('inf'), 0.0
        )
        
        a_ij = deg_row_inv_sqrt[row] * a_ij * deg_col_inv_sqrt[col]

        # Apply dropout
        if self.training and self.dropout > 0.0:
            a_ij = F.dropout(a_ij, p=self.dropout, training=True)

        return a_ij

    def message(
        self,
        x_j: Tensor,
        z_scale_i: Tensor,
        z_scale_j: Tensor,
        index: Tensor,
        num_nodes: int,
    ) -> Tensor:
        r"""Constructs messages from source nodes :math:`j` to target nodes :math:`i`.

        Args:
            x_j (torch.Tensor): Source node features of shape
                :math:`(|\mathcal{E}|, H, F_{out})`.
            z_scale_i (torch.Tensor): Scaled features for target nodes of shape
                :math:`(|\mathcal{E}|, H, F_{out})`.
            z_scale_j (torch.Tensor): Scaled features for source nodes of shape
                :math:`(|\mathcal{E}|, H, F_{out})`.
            index (torch.Tensor): Target node indices for aggregation of shape
                :math:`(|\mathcal{E}|,)`.
            num_nodes (int): Number of nodes.

        Returns:
            torch.Tensor: Messages of shape :math:`(|\mathcal{E}|, H, F_{out})`.
        """
        if self._edge_index is None:
            raise RuntimeError("edge_index not set. This should not happen.")
        a = self._edge_attention(z_scale_i, z_scale_j, self._edge_index, num_nodes)
        return a.unsqueeze(-1) * x_j

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads}, '
                f'iterations={self.iterations})')

