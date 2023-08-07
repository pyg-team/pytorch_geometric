import torch
from torch import Tensor
from torch.nn import Parameter

from torch_geometric import utils
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import Adj, OptTensor, SparseTensor
from torch_geometric.utils import is_torch_sparse_tensor, to_edge_index


class GTVConv(MessagePassing):
    r"""
    The GTVConv layer from the `"Total Variation Graph Neural Networks"
    <https://arxiv.org/abs/2211.06218>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \left(\mathbf{I} -
        \delta\mathbf{L}_\hat{\mathbf{\Gamma}}\right)\tilde{\mathbf{X}}

    where

    .. math::
        \tilde{\mathbf{X}} &= \mathbf{X} \mathbf{\Theta}

        \mathbf{L}_{\hat{\mathbf{\Gamma}}} &=
        \mathbf{D}_{\hat{\mathbf{\Gamma}}} - \hat{\mathbf{\Gamma}}

        \hat{\Gamma}_{ij} &= \frac{A_{ij}}{\max\{||\tilde{\mathbf{x}}_i-
        \tilde{\mathbf{x}}_j||_1, \epsilon\}}

    Args:
        in_channels (int):
            Size of each input sample
        out_channels (int):
            Size of each output sample.
        bias (bool):
            If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        delta_coeff (float):
            Step size for gradient descent of GTV (default: :obj:`1.0`)
        eps (float):
            Small number used to numerically stabilize the computation of
            new adjacency weights. (default: :obj:`1e-3`)

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})`,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge weights :math:`(|\mathcal{E}|)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})`
    """
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True,
                 delta_coeff: float = 1., eps: float = 1e-3, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.delta_coeff = delta_coeff
        self.eps = eps

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        glorot(self.weight)
        zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:

        # Update node features
        x = x @ self.weight

        if isinstance(edge_index, SparseTensor):
            col, row, edge_weight = edge_index.coo()
            edge_index = torch.stack((row, col), dim=0)

        elif is_torch_sparse_tensor(edge_index):
            edge_index, edge_weight = to_edge_index(edge_index)
            col, row = edge_index[0], edge_index[1]
            edge_index = torch.stack((row, col), dim=0)

        else:
            row, col = edge_index[0], edge_index[1]

        # Absolute differences between neighbouring nodes
        abs_diff = torch.abs(x[row, :] - x[col, :])
        abs_diff = abs_diff.sum(dim=1)

        # Gamma matrix
        denom = torch.clamp(abs_diff, min=self.eps)
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=x.dtype,
                                     device=edge_index.device)

        gamma_vals = edge_weight / denom

        # Laplacian L = D - A
        lap_index, lap_weight = utils.get_laplacian(edge_index, gamma_vals,
                                                    normalization=None)

        # Modified laplacian: I - delta * L
        if lap_weight is None:
            lap_weight = torch.ones((edge_index.size(1), ), dtype=x.dtype,
                                    device=edge_index.device)
        lap_weight *= -self.delta_coeff
        mod_lap_index, mod_lap_weight = utils.add_self_loops(
            lap_index, lap_weight, fill_value=1., num_nodes=x.size(0))

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index=mod_lap_index, x=x,
                             edge_weight=mod_lap_weight, size=None)

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j
