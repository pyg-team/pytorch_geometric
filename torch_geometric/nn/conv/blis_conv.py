from typing import List

import torch
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import spmm


class LazyLayer(torch.nn.Module):
    r"""A learnable, per-channel convex combination of a signal and its
    diffused version (a "laziness" parameter passed through a softmax).
    """
    def __init__(self, n: int) -> None:
        super().__init__()
        self.n = n
        self.weights = Parameter(torch.empty(2, n))
        self.reset_parameters()

    def forward(self, x: Tensor, propagated: Tensor) -> Tensor:
        s_weights = torch.softmax(self.weights, dim=0)
        inp = torch.stack((x, propagated), dim=0)
        return torch.sum(inp * s_weights.view(2, 1, self.n), dim=0)

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.weights)


def _build_wavelet_constructor(max_scale_power: int) -> Tensor:
    r"""Builds the dyadic ``W2`` wavelet-difference matrix.

    Given ``max_scale_power`` :math:`= K`, the diffusion scales are
    :math:`\{0, 2^0, 2^1, \ldots, 2^K\}`. Each band-pass wavelet is the
    difference of diffusion operators at consecutive dyadic scales,
    :math:`\mathbf{P}^{2^k} - \mathbf{P}^{2^{k+1}}`, and a final low-pass
    filter :math:`\mathbf{P}^{2^K}` is appended. Returns the
    ``[num_filters, num_diffusion + 1]`` selection matrix.
    """
    dyadic = [0] + [2**k for k in range(max_scale_power + 1)]
    num_diffusion = dyadic[-1]
    num_filters = len(dyadic)  # (len(dyadic) - 1) band-pass + 1 low-pass.

    weight = torch.zeros(num_filters, num_diffusion + 1)
    for i in range(len(dyadic) - 1):
        weight[i, dyadic[i]] = 1.0
        weight[i, dyadic[i + 1]] = -1.0
    weight[-1, dyadic[-1]] = 1.0  # low-pass filter.
    return weight


class BLISConv(MessagePassing):
    r"""The bi-Lipschitz geometric-scattering operator from the `"BLIS-Net:
    Classifying and Analyzing Signals on Graphs"
    <https://proceedings.mlr.press/v238/xu24c.html>`_ paper.

    For an input node signal, the layer computes diffusion-wavelet coefficients
    on every node using the dyadic ``W2`` wavelet bank derived from the lazy
    random walk
    :math:`\mathbf{P} = \frac{1}{2}(\mathbf{I} + \mathbf{A}\mathbf{D}^{-1})`,

    .. math::
        \mathbf{W}_k = \mathbf{P}^{2^{k-1}} - \mathbf{P}^{2^k}, \quad
        \mathbf{W}_{\text{low}} = \mathbf{P}^{2^K},

    followed by the bi-Lipschitz activation
    :math:`[\textrm{ReLU}(x), \textrm{ReLU}(-x)]`. Because the full wavelet
    frame is applied at every layer, the energy of the input signal is
    preserved; stacking :math:`m` layers therefore realizes an :math:`m`-th
    order scattering cascade whose informative output is simply the final
    layer's coefficients (no skip connections required).

    With :math:`K =` :obj:`max_scale_power`, the bank contains :math:`K + 2`
    filters. For an input with :math:`F` channels, the output has
    :math:`(K + 2) \cdot A \cdot F` channels per node, where :math:`A` is the
    number of activations (:obj:`2` for :obj:`"blis"`), available as
    :obj:`out_channels`.

    .. note::
        The layer assumes an undirected graph (as in the paper); edge
        directions are treated symmetrically by the random-walk normalization.

    Args:
        in_channels (int): Size of each input sample :math:`F`.
        max_scale_power (int, optional): The largest dyadic diffusion scale is
            :math:`2^{\texttt{max\_scale\_power}}` (:math:`K`).
            (default: :obj:`4`)
        activation (str, optional): The pointwise nonlinearity, either
            :obj:`"blis"` for the bi-Lipschitz :math:`[\textrm{ReLU}(x),
            \textrm{ReLU}(-x)]` map, or :obj:`"identity"` for the linear
            scattering transform. (default: :obj:`"blis"`)
        trainable_laziness (bool, optional): If set to :obj:`True`, uses a
            learnable laziness parameter in the diffusion.
            (default: :obj:`False`)
        trainable_scales (bool, optional): If set to :obj:`True`, the
            wavelet-difference matrix becomes a learnable parameter.
            (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F)` (or :math:`(|\mathcal{V}|)`
          for a scalar signal),
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge weights :math:`(|\mathcal{E}|)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|,` :obj:`out_channels`
          :math:`)`
    """
    def __init__(
        self,
        in_channels: int,
        max_scale_power: int = 4,
        activation: str = 'blis',
        trainable_laziness: bool = False,
        trainable_scales: bool = False,
        **kwargs,
    ) -> None:
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        if activation not in ('blis', 'identity'):
            raise ValueError(f"Invalid activation '{activation}' "
                             f"(expected 'blis' or 'identity')")

        self.in_channels = in_channels
        self.max_scale_power = max_scale_power
        self.activation = activation
        self.trainable_laziness = trainable_laziness
        self.trainable_scales = trainable_scales

        weight = _build_wavelet_constructor(max_scale_power)
        self.num_diffusion = weight.size(1) - 1
        self.num_filters = weight.size(0)
        self.num_activations = 2 if activation == 'blis' else 1
        self.out_channels = (self.num_filters * self.num_activations *
                             in_channels)

        if trainable_scales:
            self.wavelet_constructor = Parameter(weight)
        else:
            self.register_buffer('wavelet_constructor', weight)

        self.lazy_layer = LazyLayer(
            in_channels) if trainable_laziness else None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        super().reset_parameters()
        if self.lazy_layer is not None:
            self.lazy_layer.reset_parameters()
        if isinstance(self.wavelet_constructor, Parameter):
            with torch.no_grad():
                weight = _build_wavelet_constructor(self.max_scale_power)
                self.wavelet_constructor.copy_(
                    weight.to(self.wavelet_constructor.device))

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        if x.dim() == 1:
            x = x.view(-1, 1)

        # T = A D^-1 acts as T x = A (x / deg), so a single unnormalized
        # aggregation of an all-ones signal yields the (weighted) degrees. This
        # keeps the layer agnostic to the `edge_index` / `SparseTensor` format.
        ones = x.new_ones((x.size(self.node_dim), 1))
        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        deg = self.propagate(edge_index, x=ones, edge_weight=edge_weight)
        deg_inv = deg.pow(-1)
        deg_inv.masked_fill_(deg_inv == float('inf'), 0)

        avgs: List[Tensor] = [x]
        cur = x
        for _ in range(self.num_diffusion):
            propagated = self.propagate(edge_index, x=cur * deg_inv,
                                        edge_weight=edge_weight)
            if self.lazy_layer is not None:
                cur = self.lazy_layer(cur, propagated)
            else:
                cur = 0.5 * (cur + propagated)
            avgs.append(cur)

        # Select dyadic differences: [num_filters, num_nodes, F].
        levels = torch.stack(avgs, dim=0)
        coeffs = torch.einsum('ij,jnf->inf', self.wavelet_constructor, levels)

        if self.activation == 'blis':
            coeffs = torch.stack([coeffs.relu(), (-coeffs).relu()], dim=0)
        else:
            coeffs = coeffs.unsqueeze(0)

        # [A, num_filters, num_nodes, F] -> [num_nodes, out_channels].
        num_nodes = coeffs.size(2)
        return coeffs.permute(2, 0, 1, 3).reshape(num_nodes, -1)

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'out_channels={self.out_channels}, '
                f'K={self.max_scale_power}, activation={self.activation})')
