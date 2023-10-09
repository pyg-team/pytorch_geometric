from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import ModuleList, Sequential

from torch_geometric.nn.conv import PNAConv
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.resolver import activation_resolver


class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)


class GMNConv(PNAConv):
    r"""The Graph Mixer convolution operator
    from the `"The Graph Mixer Networks"
    <https://arxiv.org/abs/2301.12493>`_ paper

    .. math::
        \mathbf{x}_i^{\prime} = \Mix \left(
        \mathbf{x}_i, \underset{j \in \mathcal{N}(i)}{\bigoplus}
        \left( \mathbf{x}_i, \mathbf{x}_j \right)
        \right)

    with

    .. math::

        \Mix = \MLP_{2}(\LayerNorm((\MLP_{1}
        ((\LayerNorm(\x))^{T}))^{T} + \x)) + \x

    and

    .. math::
        \bigoplus = \underbrace{\begin{bmatrix}
            1 \\
            S(\mathbf{D}, \alpha=1) \\
            S(\mathbf{D}, \alpha=-1)
        \end{bmatrix} }_{\text{scalers}}
        \otimes \underbrace{\begin{bmatrix}
            \mu \\
            \sigma \\
            \max \\
            \min
        \end{bmatrix}}_{\text{aggregators}},

    where :math:`\gamma_{\mathbf{\Theta}}` denotes MLPs.

    .. note::

        For an example of using :obj:`GMNConv`, see `examples/gmn.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/
        examples/gmn.py>`_.

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        out_channels (int): Size of each output sample.
        aggregators (List[str]): Set of aggregation function identifiers,
            namely :obj:`"sum"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`,
            :obj:`"var"` and :obj:`"std"`.
        scalers (List[str]): Set of scaling function identifiers, namely
            :obj:`"identity"`, :obj:`"amplification"`,
            :obj:`"attenuation"`, :obj:`"linear"` and
            :obj:`"inverse_linear"`.
        deg (torch.Tensor): Histogram of in-degrees of nodes in the training
            set, used by scalers to normalize.
        edge_dim (int, optional): Edge feature dimensionality (in case
            there are any). (default :obj:`None`)
        towers (int, optional): Number of towers (default: :obj:`1`).
        post_layers (int, optional): Number of transformation layers after
            aggregation (default: :obj:`1`).
        divide_input (bool, optional): Whether the input features should
            be split between towers or not (default: :obj:`False`).
        act (str or callable, optional): Pre- and post-layer activation
            function to use. (default: :obj:`"relu"`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        train_norm (bool, optional): Whether normalization parameters
            are trainable. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})`,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge features :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})`
    """
    def __init__(self, in_channels: int, out_channels: int,
                 aggregators: List[str], scalers: List[str], deg: Tensor,
                 edge_dim: Optional[int] = None, towers: int = 1,
                 post_layers: int = 1, divide_input: bool = False,
                 act: Union[str, Callable, None] = "relu",
                 act_kwargs: Optional[Dict[str, Any]] = None, **kwargs):

        super().__init__(in_channels, out_channels, aggregators, scalers, deg,
                         edge_dim, towers, divide_input, **kwargs)

        self.post_nns = ModuleList()
        for _ in range(towers):

            in_channels = (len(aggregators) * len(scalers) + 1) * self.F_in
            modules = [Linear(in_channels, self.F_out)]

            for _ in range(post_layers - 1):
                x = self.F_out
                modules += [nn.LayerNorm([x])]
                modules += [Transpose(1, 2)]
                modules += [activation_resolver(act, **(act_kwargs or {}))]
                modules += [Linear(self.F_out, self.F_out)]
                modules += [Transpose(1, 2)]
                modules += x
                modules += [nn.LayerNorm([self.F_out])]
                modules += [activation_resolver(act, **(act_kwargs or {}))]
                modules += [Linear(self.F_out, self.F_out)]
            self.post_nns.append(Sequential(*modules))

        self.lin = Linear(out_channels, out_channels)

        self.reset_parameters()

    def message(self, x_j: Tensor) -> Tensor:
        return x_j
