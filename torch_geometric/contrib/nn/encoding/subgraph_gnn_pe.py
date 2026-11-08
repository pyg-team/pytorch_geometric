import copy
import inspect
from typing import Any, Callable, Dict, Optional, Union

import torch
from torch import Tensor
from torch.nn import ModuleList

from torch_geometric.nn.conv import GINConv
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.models import MLP
from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)
from torch_geometric.typing import OptTensor


class SubgraphGNNPE(torch.nn.Module):
    r"""The subgraph GNN positional encoding from the `"Relational Graph
    Transformer" <https://arxiv.org/abs/2505.10960>`_ paper.

    A lightweight GNN is applied to (a batch of) small sampled subgraphs,
    using random node features as input:

    .. math::
        \mathbf{H}_{\textrm{PE}} = \textrm{GNN}(\mathbf{A},
        \mathbf{Z}), \quad \mathbf{Z} \sim \mathcal{N}(\mathbf{0},
        \mathbf{I}) \in \mathbb{R}^{N \times d_{\textrm{init}}}

    Random node features break structural symmetries and increase the
    expressiveness of the GNN (see the `"Random Features Strengthen Graph
    Neural Networks" <https://arxiv.org/abs/2002.03155>`_ paper).
    Since :math:`\mathbf{Z}` is resampled in every forward pass, the encoding
    remains permutation equivariant in expectation.
    No precomputation (*e.g.*, of Laplacian eigenvectors) is required.

    Following the reference implementation, :math:`\mathbf{Z}` is linearly
    projected to :obj:`hidden_channels`, followed by :obj:`num_layers`
    residual blocks of :class:`~torch_geometric.nn.conv.GINConv`,
    normalization and activation, and a final linear projection to
    :obj:`out_channels`.
    The default arguments match the reference implementation.

    .. note::

        Random features are resampled in evaluation mode as well, following
        the reference implementation.
        Pass :obj:`x` to :meth:`forward` for deterministic outputs.

    Args:
        out_channels (int): Size of each output positional encoding.
        hidden_channels (int, optional): Size of each hidden sample.
            If set to :obj:`None`, will use :obj:`out_channels // 4`.
            (default: :obj:`None`)
        num_layers (int, optional): Number of message passing layers.
            (default: :obj:`4`)
        in_channels (int, optional): Size :math:`d_{\textrm{init}}` of the
            random input features. (default: :obj:`1`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str, optional): The normalization function to use, both inside
            the MLP of each :class:`~torch_geometric.nn.conv.GINConv` and
            after each layer. (default: :obj:`"batch_norm"`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
            of each :class:`~torch_geometric.nn.conv.GINConv` will be a
            trainable parameter. (default: :obj:`True`)

    Shapes:
        - **input:**
          edge indices :math:`(2, |\mathcal{E}|)`,
          node features :math:`(|\mathcal{V}|, d_{\textrm{init}})`
          *(optional)*,
          batch vector :math:`(|\mathcal{V}|)` *(optional)*
        - **output:** positional encodings
          :math:`(|\mathcal{V}|, F_{out})`
    """
    def __init__(
        self,
        out_channels: int,
        hidden_channels: Optional[int] = None,
        num_layers: int = 4,
        in_channels: int = 1,
        act: Union[str, Callable, None] = 'relu',
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Optional[str] = 'batch_norm',
        norm_kwargs: Optional[Dict[str, Any]] = None,
        train_eps: bool = True,
    ):
        super().__init__()

        if hidden_channels is None:
            hidden_channels = out_channels // 4

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.act = activation_resolver(act, **(act_kwargs or {}))

        self.lin_in = Linear(in_channels, hidden_channels,
                             weight_initializer='glorot',
                             bias_initializer='zeros')

        norm_layer = normalization_resolver(norm, hidden_channels,
                                            **(norm_kwargs or {}))
        if norm_layer is None:
            norm_layer = torch.nn.Identity()
        norm_params = inspect.signature(norm_layer.forward).parameters
        self.supports_norm_batch = 'batch' in norm_params

        self.convs = ModuleList()
        self.norms = ModuleList()
        for _ in range(num_layers):
            mlp = MLP(
                [hidden_channels, 2 * hidden_channels, hidden_channels],
                act=self.act,
                norm=norm,
                norm_kwargs=norm_kwargs,
            )
            self.convs.append(GINConv(mlp, train_eps=train_eps))
            self.norms.append(copy.deepcopy(norm_layer))

        self.lin_out = Linear(hidden_channels, out_channels,
                              weight_initializer='glorot',
                              bias_initializer='zeros')

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.lin_in.reset_parameters()
        for conv, norm in zip(self.convs, self.norms):
            conv.reset_parameters()
            if hasattr(norm, 'reset_parameters'):
                norm.reset_parameters()
        self.lin_out.reset_parameters()

    def forward(
        self,
        edge_index: Tensor,
        num_nodes: int,
        x: OptTensor = None,
        batch: OptTensor = None,
        batch_size: Optional[int] = None,
    ) -> Tensor:
        r"""Forward pass.

        Args:
            edge_index (torch.Tensor): The edge indices of the (batched)
                subgraphs.
            num_nodes (int): The total number of nodes :math:`N`, including
                isolated ones.
            x (torch.Tensor, optional): Node features of shape
                :obj:`[num_nodes, in_channels]` to use instead of random ones.
                (default: :obj:`None`)
            batch (torch.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each node to a specific subgraph.
                Only needs to be passed in case the underlying normalization
                layers require the :obj:`batch` information.
                (default: :obj:`None`)
            batch_size (int, optional): The number of subgraphs :math:`B`.
                Automatically calculated if not given.
                Only needs to be passed in case the underlying normalization
                layers require the :obj:`batch` information.
                (default: :obj:`None`)
        """
        if x is None:
            weight = self.lin_in.weight
            x = torch.randn(num_nodes, self.in_channels, device=weight.device,
                            dtype=weight.dtype)
        elif x.size(0) != num_nodes:
            raise ValueError(f"Expected 'x' to hold features for {num_nodes} "
                             f"nodes (got {x.size(0)})")

        x = self.lin_in(x)
        for conv, norm in zip(self.convs, self.norms):
            h = conv(x, edge_index)
            if self.supports_norm_batch:
                h = norm(h, batch, batch_size)
            else:
                h = norm(h)
            if self.act is not None:
                h = self.act(h)
            x = x + h
        return self.lin_out(x)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.out_channels}, '
                f'hidden_channels={self.hidden_channels}, '
                f'num_layers={self.num_layers})')
