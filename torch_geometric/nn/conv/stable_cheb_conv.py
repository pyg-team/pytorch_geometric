from typing import Optional

import torch
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import OptTensor
from torch_geometric.utils import get_laplacian


class StableChebConv(MessagePassing):
    r"""The Stable Chebyshev spectral graph convolutional operator from the
    `"Return of ChebNet: Understanding and Improving an Overlooked GNN on
    Long-Range Tasks" <https://arxiv.org/abs/2506.07624>`_ paper(NeurIPS 2025).

    ChebNet is recast as a stable, non-dissipative dynamical system and
    discretised via a forward Euler step:

    .. math::
        \mathbf{X}^{\prime} = \mathbf{X} + \varepsilon \sum_{k=1}^{K}
        \mathbf{Z}^{(k)} \cdot \mathbf{A}^{(k)}

    where the antisymmetric effective weight at order :math:`k` is

    .. math::
        \mathbf{A}^{(k)} = \mathbf{W}^{(k)} - {\mathbf{W}^{(k)}}^{\top}
        - \gamma \mathbf{I}

    and :math:`\mathbf{Z}^{(k)}` is computed recursively by

    .. math::
        \mathbf{Z}^{(1)} &= \mathbf{X}

        \mathbf{Z}^{(2)} &= \mathbf{\hat{L}} \cdot \mathbf{X}

        \mathbf{Z}^{(k)} &= 2 \cdot \mathbf{\hat{L}} \cdot
        \mathbf{Z}^{(k-1)} - \mathbf{Z}^{(k-2)}

    and :math:`\mathbf{\hat{L}}` denotes the scaled and normalized Laplacian
    :math:`\frac{2\mathbf{L}}{\lambda_{\max}} - \mathbf{I}`.

    The antisymmetry of :math:`\mathbf{A}^{(k)}` forces the layer-wise
    Jacobian to have purely imaginary eigenvalues, yielding
    :math:`\|\mathbf{J}^{(l)}\|_2 = 1 + \mathcal{O}(\varepsilon^2)` and
    preventing the exponential growth or decay that destabilises vanilla
    ChebNet at large filter orders :math:`K`.

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        out_channels (int): Size of each output sample. The antisymmetric
            weight matrices are always :math:`d \times d` where
            :math:`d = \text{out\_channels}`. An input projection is added
            automatically when :obj:`in_channels != out_channels`.
        K (int): Chebyshev filter size :math:`K`.
        epsilon (float, optional): Forward Euler step size
            :math:`\varepsilon`. Smaller values improve numerical stability;
            larger values accelerate convergence. Typical sweep range
            :obj:`[0.1, 1.0]`. (default: :obj:`0.5`)
        gamma (float, optional): Dissipative regularisation :math:`\gamma`
            subtracted from the diagonal of each antisymmetric weight.
            Setting :obj:`gamma=0` recovers a purely conservative,
            energy-preserving system. (default: :obj:`0.1`)
        normalization (str, optional): The normalization scheme for the graph
            Laplacian (default: :obj:`"sym"`):

            1. :obj:`None`: No normalization
               :math:`\mathbf{L} = \mathbf{D} - \mathbf{A}`

            2. :obj:`"sym"`: Symmetric normalization
               :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2}
               \mathbf{A} \mathbf{D}^{-1/2}`

            3. :obj:`"rw"`: Random-walk normalization
               :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1} \mathbf{A}`

            :obj:`\lambda_max` should be a :class:`torch.Tensor` of size
            :obj:`[num_graphs]` in a mini-batch scenario and a
            scalar/zero-dimensional tensor when operating on single graphs.
            You can pre-compute :obj:`lambda_max` via the
            :class:`torch_geometric.transforms.LaplacianLambdaMax` transform.
        bias (bool, optional): If set to :obj:`False`, the layer will not
            learn an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})`,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge weights :math:`(|\mathcal{E}|)` *(optional)*,
          batch vector :math:`(|\mathcal{V}|)` *(optional)*,
          maximum :obj:`lambda` value :math:`(|\mathcal{G}|)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})`
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int,
        epsilon: float = 0.5,
        gamma: float = 0.1,
        normalization: Optional[str] = 'sym',
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        assert K > 0, "K must be a positive integer"
        assert normalization in [None, 'sym', 'rw'], "Invalid normalization"
        assert epsilon > 0.0, "epsilon must be strictly positive"
        assert gamma >= 0.0, "gamma must be non-negative"

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.epsilon = epsilon
        self.gamma = gamma
        self.normalization = normalization

        if in_channels != out_channels:
            self.in_proj: Optional[Linear] = Linear(
                in_channels,
                out_channels,
                bias=False,
                weight_initializer='glorot',
            )
        else:
            self.in_proj = None

        self.weights = torch.nn.ParameterList([
            Parameter(torch.empty(out_channels, out_channels))
            for _ in range(K)
        ])

        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        super().reset_parameters()
        for w in self.weights:
            torch.nn.init.orthogonal_(w)
            with torch.no_grad():
                w.mul_(0.01)
        if self.in_proj is not None:
            self.in_proj.reset_parameters()
        if self.bias is not None:
            zeros(self.bias)

    def _antisymmetric(self, w: Tensor) -> Tensor:
        asym = w - w.t()
        if self.gamma != 0.0:
            asym = asym - self.gamma * torch.eye(
                w.size(0),
                dtype=w.dtype,
                device=w.device,
            )
        return asym

    def __norm__(
        self,
        edge_index: Tensor,
        num_nodes: Optional[int],
        edge_weight: OptTensor,
        normalization: Optional[str],
        lambda_max: OptTensor = None,
        dtype=None,
        batch: OptTensor = None,
    ):

        edge_index, edge_weight = get_laplacian(
            edge_index,
            edge_weight,
            normalization,
            dtype,
            num_nodes,
        )
        assert edge_weight is not None

        if lambda_max is None:
            lambda_max = 2.0 * edge_weight.max()
        elif not isinstance(lambda_max, Tensor):
            lambda_max = torch.tensor(
                lambda_max,
                dtype=dtype,
                device=edge_index.device,
            )
        assert lambda_max is not None

        if batch is not None and lambda_max.numel() > 1:
            lambda_max = lambda_max[batch[edge_index[0]]]

        edge_weight = (2.0 * edge_weight) / lambda_max
        edge_weight.masked_fill_(edge_weight == float('inf'), 0)

        loop_mask = edge_index[0] == edge_index[1]
        edge_weight[loop_mask] -= 1

        return edge_index, edge_weight

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: OptTensor = None,
        batch: OptTensor = None,
        lambda_max: OptTensor = None,
    ) -> Tensor:

        x = self.in_proj(x) if self.in_proj is not None else x
        edge_index, norm = self.__norm__(
            edge_index,
            x.size(self.node_dim),
            edge_weight,
            self.normalization,
            lambda_max,
            dtype=x.dtype,
            batch=batch,
        )

        Tx_0 = x
        Tx_1 = x
        update = Tx_0 @ self._antisymmetric(self.weights[0])

        if self.K > 1:
            Tx_1 = self.propagate(edge_index, x=Tx_0, norm=norm)
            update = update + Tx_1 @ self._antisymmetric(self.weights[1])

        for k in range(2, self.K):
            Tx_2 = self.propagate(edge_index, x=Tx_1, norm=norm)
            Tx_2 = 2.0 * Tx_2 - Tx_0
            update = update + Tx_2 @ self._antisymmetric(self.weights[k])
            Tx_0, Tx_1 = Tx_1, Tx_2

        out = x + self.epsilon * update

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j: Tensor, norm: Tensor) -> Tensor:
        return norm.view(-1, 1) * x_j

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'{self.in_channels}, {self.out_channels}, '
                f'K={self.K}, '
                f'epsilon={self.epsilon}, '
                f'gamma={self.gamma}, '
                f'normalization={self.normalization!r})')
