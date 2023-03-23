from typing import Dict, Optional

import numpy as np
import torch
from torch import Tensor

import torch_geometric as pyg
from torch_geometric.contrib.utils import Fiber
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj
from torch_geometric.utils import scatter, softmax


def _degree_to_dim(degree: int) -> int:
    return 2 * degree + 1


class RadialProfile(torch.nn.Module):
    """
    Radial profile function.
    Outputs weights used to weigh basis matrices in order to get convolution kernels.
    In TFN notation: $R^{l,k}$
    In SE(3)-Transformer notation: $\phi^{l,k}$

    Note:
        In the original papers, this function only depends on relative node distances ||x||.
        Here, we allow this function to also take as input additional invariant edge features.
        This does not break equivariance and adds expressive power to the model.

    Diagram:
        invariant edge features (node distances included) ───> MLP layer (shared across edges) ───> radial weights
    """
    def __init__(
        self,
        num_freq: int,
        channels_in: int,
        channels_out: int,
        edge_dim: int = 1,
        mid_dim: int = 32,
        use_layer_norm: bool = False,
    ):
        """
        :param num_freq:         Number of frequencies
        :param channels_in:      Number of input channels
        :param channels_out:     Number of output channels
        :param edge_dim:         Number of invariant edge features (input to the radial function)
        :param mid_dim:          Size of the hidden MLP layers
        :param use_layer_norm:   Apply layer normalization between MLP layers
        """
        super().__init__()
        modules = [
            torch.nn.Linear(edge_dim, mid_dim),
            torch.nn.LayerNorm(mid_dim) if use_layer_norm else None,
            torch.nn.ReLU(),
            torch.nn.Linear(mid_dim, mid_dim),
            torch.nn.LayerNorm(mid_dim) if use_layer_norm else None,
            torch.nn.ReLU(),
            torch.nn.Linear(mid_dim, num_freq * channels_in * channels_out,
                            bias=False),
        ]

        self.net = torch.nn.Sequential(*[m for m in modules if m is not None])

    def forward(self, features: Tensor) -> Tensor:
        return self.net(features)


class SE3ConvUnit(torch.nn.Module):
    def __init__(
        self,
        freq_sum: int,
        channels_in: int,
        channels_out: int,
        edge_dim: int,
        use_layer_norm: bool,
    ):
        super().__init__()
        self.freq_sum = freq_sum
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.radial_func = RadialProfile(
            num_freq=freq_sum,
            channels_in=channels_in,
            channels_out=channels_out,
            edge_dim=edge_dim,
            use_layer_norm=use_layer_norm,
        )

    def forward(self, features: Tensor, invariant_edge_feats: Tensor,
                basis: Tensor):
        num_edges = features.shape[0]
        in_dim = features.shape[2]
        radial_weights = self.radial_func(invariant_edge_feats)
        radial_weights = radial_weights.view(-1, self.channels_out,
                                             self.channels_in * self.freq_sum)

        basis_view = basis.view(num_edges, in_dim, -1)
        tmp = (features @ basis_view).view(num_edges, -1, basis.shape[-1])
        return radial_weights @ tmp


class SE3Conv(MessagePassing):
    r"""The SE(3) Equivariant graph convolutional operator from the
    `"Tensor field networks: Tensor field networks: Rotation- and
    translation-equivariant neural networks for 3D point clouds"
    <https://arxiv.org/abs/1802.08219>`_ paper. This convolution can map an arbitrary
    input Fiber to an arbitrary output Fiber, while preserving equivariance.
    Features of different degrees interact together to produce output features.

    .. math::
        \mathbf{x}^{\prime, \ell}_i =
        \sum_{j \in \mathcal{N}(i)} \sum_{k \ge 0}
        \Theta^{\ell,k} (\mathbf{e}_{i,j}) \mathbf{x}_j^k

    Where :math:`\mathbf{x}^{\prime, \ell}` are the output node features of x
    that are type-:math:`\ell`, and :math:`\mathbf{e}_{i,j}` are the edge features.

    To maintain rotational equivariance, each
    :math:`\Theta^{\ell,k}()\mathbf{e}_{i,j}` must be a linear combination of
    equivariant bases so that

    .. math::
        \Theta^{\ell,k}(\mathbf{e}_{i,j}) =
        \sum_{J=|k-\ell|}^{k+\ell}
        \varphi^{\ell,k}_J(e^0_{i,j}) \Theta^{\ell,k}_J(e_{i,j})

    Where $\varphi^{\ell,k}_J$ is a learnable function that takes in the type-0
    (rotation invariant) edge features (including the edge length) and outputs
    a real coefficient, and

    .. math::
        \Theta^{\ell,k}_J(e_{i,j}) =
        \sum_{-J}^{J}
        Y_{J,m}(\mathbf{x_j}^\text{pos} - \mathbf{x}_i^{\text{pos}})
        \mathbf{Q}^{\ell,k}_{J,m}


    Here, :math:`\mathbf{Q}^{\ell,k}_{J,m}` is the Clebsch-Gordon matrix of
    shape :math:`(2\ell + 1) \times (2k + 1)` and
    :math:`Y_{J,m}` is the :math:`m^{\text{th}}` dimension of the
    :math:`J^{\text{th}}` spherical harmonic
    :math:`Y_J : \mathbb{R}^3 \to \mathbb{R}^{2J+1}`.

    If `self_interaction` is True

    .. math::
        \mathbf{x}^{\prime, \ell}_i =
        w^{\ell,\ell}x^{\ell}_i +
        \sum_{j \in \mathcal{N}(i)} \sum_{k \ge 0}
        \Theta^{\ell,k} (\mathbf{e}_{i,j}) \mathbf{x}_j^k


    Note 1:
        The option is given to not pool the output. This means that the convolution
        sum over neighbors will not be done, and the returned features will be edge
        features instead of node features.

    Note 2:
        Unlike the original paper and implementation, this convolution can handle
        edge feature of degree greater than 0. Input edge features are concatenated
        with input source node features before the kernel is applied.

    Args:
        fiber_in (:class:`torch_geometric.contrib.utils.Fiber`): Fiber
            describing the input features
        fiber_out (:class:`torch_geometric.contrib.utils.Fiber`): Fiber
            describing the output features
        fiber_edge (:class:`torch_geometric.contrib.utils.Fiber`): Fiber
            describing the edge features
        use_layer_norm (bool): Apply layer normalization between MLP layers
        self_interaction (bool, optional): Apply self-interaction of nodes
            (default: :obj:`False`)
        aggregate (bool, optional): If False, returns edge features without
            aggregation.
            (default: :obj:`True`)
    """
    def __init__(
        self,
        fiber_in,
        fiber_out,
        fiber_edge,
        use_layer_norm,
        self_interaction: bool = False,
        aggregate: bool = True,
    ):
        super().__init__(node_dim=0, aggr="add")
        self.fiber_in = fiber_in
        self.fiber_out = fiber_out
        self.fiber_edge = fiber_edge
        self.self_interaction = self_interaction
        self.aggr = aggregate

        common_args = dict(edge_dim=fiber_edge[0],
                           use_layer_norm=use_layer_norm)

        self.conv = torch.nn.ModuleDict()
        for (degree_in,
             channels_in), (degree_out,
                            channels_out) in (self.fiber_in * self.fiber_out):
            dict_key = f"{degree_in},{degree_out}"
            channels_in_new = channels_in + fiber_edge[degree_in] * (degree_in
                                                                     > 0)
            sum_freq = _degree_to_dim(min(degree_in, degree_out))
            self.conv[dict_key] = SE3ConvUnit(sum_freq, channels_in_new,
                                              channels_out, **common_args)

        if self_interaction:
            self.to_kernel_self = torch.nn.ParameterDict()
            for degree_out, channels_out in fiber_out:
                if degree_out in fiber_in:
                    self.to_kernel_self[str(degree_out)] = torch.nn.Parameter(
                        torch.randn(channels_out, fiber_in[degree_out]) /
                        np.sqrt(fiber_in[degree_out]))

    def forward(
        self,
        node_feats: Dict[str, Tensor],
        edge_feats: Dict[str, Tensor],
        edge_index,
        basis: Dict[str, Tensor],
    ):
        # TODO: Check and make sure the dists are being passed here
        invariant_edge_feats = edge_feats["0"].squeeze(-1)
        src, dst = edge_index
        out = {}
        in_features = []

        # Fetch all input features from edge and node features
        for degree_in in self.fiber_in.degrees:
            src_node_features = node_feats[str(degree_in)][src]
            if degree_in > 0 and str(degree_in) in edge_feats:
                # Handle edge features of any type by concatenating them to node features
                src_node_features = torch.cat(
                    [src_node_features, edge_feats[str(degree_in)]], dim=1)
            in_features.append(src_node_features)

        # x^l = out[str(degree_out)]
        for degree_out in self.fiber_out.degrees:
            out_feature = 0

            # \sum_{k \ge 0}
            for degree_in, feature in zip(self.fiber_in.degrees, in_features):
                dict_key = f"{degree_in},{degree_out}"  # j,k
                basis_used = basis.get(dict_key, None)

                # \Omega^{l,k}(e_{i,j}) x^k
                out_feature += self.conv[dict_key](feature,
                                                   invariant_edge_feats,
                                                   basis_used)

            if self.aggr:
                # \sum_{j \in N(i)}
                out[str(degree_out)] = pyg.utils.scatter(
                    out_feature, dst, reduce="sum")
            else:
                out[str(degree_out)] = out_feature

        # TODO: Handle this for self.aggr = False
        if self.self_interaction:
            for degree_out, kernel_self in self.to_kernel_self.items():
                out[str(degree_out)] = (
                    out[str(degree_out)] +
                    kernel_self @ node_feats[str(degree_out)])

        return out


class SE3GATConv(MessagePassing):
    r"""SE(3) Equivariant graph attentional operator from the `"SE(3)-Transformers: 3D Roto-Translation Equivariant Attention Networks"
    <https://arxiv.org/abs/2006.10503>`_ paper)

    .. math::
        \mathbf{x}^{\prime,\ell}_i = \Theta^{\ell,\ell}_V\mathbf{x}_i^{\ell} +
        \sum_{j \in \mathcal{N}(i)} \sum_{k \ge 0}
        \alpha_{i,j} \Theta_V^{\ell,k}(\mathbf{e}_{i,j})\mathbf{x}^k_j

    Where the attention weights :math:`\alpha_{i,j}` are a function of the queries
    :math:`\mathbf{q}_i` and keys :math:`\mathbf{k}_{i,j}`
    .. math::
        \alpha_{i,j} = \frac{\exp( \mathbf{q}_i^{\top} \mathbf{k}_{i,j} )}
            {\sum_{j' \in \mathcal{N}} \exp( \mathbf{q}_i^{\top} \mathbf{k}_{i,j} )}

    ..math::
        \mathbf{q}_i = \bigoplus_{\ell \ge 0} \sum_{k \ge 0} \Theta_Q^{\ell, k}x^k_i

    ..math::
        \mathbf{k}_{i,j} = \bigoplus_{\ell \ge 0} \sum_{k \ge 0} \Theta_K^{\ell, k}(e_{i,j})x^k_j

    Where :math:`\bigoplus` is vector concatenation.

    The parameter matrices :math:`\Theta` are constructed as in :class:`torch_geometric.contrib.utils.SE3Conv`.


    Args:
        fiber_in (:class:`torch_geometric.contrib.utils.Fiber`) Fiber describing the input features
        fiber_out (:class:`torch_geometric.contrib.utils.Fiber`) Fiber describing the output features
        fiber_edge (:class:`torch_geometric.contrib.utils.Fiber`) Fiber describing the edge features,
            including the node distance
        num_heads (int, optional): Number of attention heads.
            (default: :obj:`4`)
        channels_div (int, optional): Divide the channels by this integer for computing values
            (default :obj:`2`)
        use_layer_norm (bool, optional): Apply later normalization between MLP layers

    Shapes:
        - **input:**
        - **output:**
    """
    def __init__(
        self,
        fiber_in: Fiber,
        fiber_out: Fiber,
        fiber_edge: Optional[Fiber] = Fiber({}),
        num_heads: int = 4,
        channels_div: int = 2,
        use_layer_norm: bool = False,
    ):
        super().__init__(node_dim=0, aggr='add')
        self.num_heads = num_heads

        if fiber_edge is None:
            fiber_edge = Fiber({})
        self.fiber_in = fiber_in

        # value_fiber has same structure as fiber_out but #channels divided by 'channels_div'
        self.value_fiber = Fiber([(degree, channels // channels_div)
                                  for degree, channels in fiber_out])

        # key_query_fiber has the same structure as fiber_out, but only degrees which are in in_fiber
        # (queries are merely projected, hence degrees have to match input)
        self.key_query_fiber = Fiber([(fe.degree, fe.channels)
                                      for fe in self.value_fiber
                                      if fe.degree in fiber_in.degrees])

        self.to_key_value = SE3Conv(
            fiber_in,
            self.value_fiber + self.key_query_fiber,
            aggregate=False,
            fiber_edge=fiber_edge,
            use_layer_norm=use_layer_norm,
        )
        self.to_query = SE3Linear(fiber_in, self.key_query_fiber)
        # self.attention = SE3GATConv(num_heads, self.key_query_fiber, self.value_fiber)
        self.project = SE3Linear(self.value_fiber + fiber_in, fiber_out)

    def forward(
        self,
        node_features: Dict[str, Tensor],
        edge_features: Dict[str, Tensor],
        edge_index: Adj,
        basis: Dict[str, Tensor],
    ):
        fused_key_value = self.to_key_value(node_features, edge_features,
                                            edge_index, basis)
        key, value = self._get_key_value_from_fused(fused_key_value)
        query = self.to_query(node_features)
        key = self.key_query_fiber.to_attention_heads(key, self.num_heads)
        query = self.key_query_fiber.to_attention_heads(query, self.num_heads)

        edge_weights = self.edge_updater(edge_index, key=key,
                                         query=(query, query))

        z = {}
        for degree, channels in self.value_fiber:
            v = value[str(degree)].view(-1, self.num_heads,
                                        channels // self.num_heads,
                                        _degree_to_dim(degree))
            agg = self.propagate(edge_index, value=v,
                                 edge_weights=edge_weights)
            z[str(degree)] = agg.view(-1, channels,
                                      _degree_to_dim(degree))  # merge heads

        z_concat = self._aggregate_residual(node_features, z, "cat")
        return self.project(z_concat)

    def message(self, value, edge_weights):
        return edge_weights * value

    def edge_update(self, key, query_j, index, ptr) -> Tensor:
        alpha = (query_j * key).sum(-1)
        alpha = softmax(alpha, index, ptr)
        return alpha.reshape(-1, self.num_heads, 1, 1)

    def _get_key_value_from_fused(self, fused_key_value):
        """Extract keys and queries features from fused features"""
        if isinstance(fused_key_value, Tensor):
            # Previous layer was a fully fused convolution
            value, key = torch.chunk(fused_key_value, chunks=2, dim=-2)
        else:
            key, value = {}, {}
            for degree, feat in fused_key_value.items():
                if int(degree) in self.fiber_in.degrees:
                    value[degree], key[degree] = torch.chunk(
                        feat, chunks=2, dim=-2)
                else:
                    value[degree] = feat

        return key, value

    def _aggregate_residual(self, feats1, feats2, method: str):
        """Add or concatenate two fiber features together. If degrees don't match, will use the ones of feats2."""
        if method in ["add", "sum"]:
            return {
                k: (v + feats1[k]) if k in feats1 else v
                for k, v in feats2.items()
            }
        elif method in ["cat", "concat"]:
            return {
                k: torch.cat([v, feats1[k]], dim=1) if k in feats1 else v
                for k, v in feats2.items()
            }
        else:
            raise ValueError("Method must be add/sum or cat/concat")


class SE3Linear(torch.nn.Module):
    """
    Graph Linear SE(3)-equivariant layer, equivalent to a 1x1 convolution.
    Maps a fiber to a fiber with the same degrees (channels may be different).
    No interaction between degrees, but interaction between channels.

    type-0 features (C_0 channels) ────> Linear(bias=False) ────> type-0 features (C'_0 channels)
    type-1 features (C_1 channels) ────> Linear(bias=False) ────> type-1 features (C'_1 channels)
                                                 :
    type-k features (C_k channels) ────> Linear(bias=False) ────> type-k features (C'_k channels)
    """
    def __init__(self, fiber_in: Fiber, fiber_out: Fiber):
        super().__init__()
        self.weights = torch.nn.ParameterDict({
            str(degree_out): torch.nn.Parameter(
                torch.randn(channels_out, fiber_in[degree_out]) /
                np.sqrt(fiber_in[degree_out]))
            for degree_out, channels_out in fiber_out
        })

    def forward(self, features: Dict[str, Tensor], *args,
                **kwargs) -> Dict[str, Tensor]:
        return {
            degree: self.weights[degree] @ features[degree]
            for degree, weight in self.weights.items()
        }


class SE3Norm(torch.nn.Module):
    """
    Norm-based SE(3)-equivariant nonlinearity.

                 ┌──> feature_norm ──> LayerNorm() ──> ReLU() ──┐
    feature_in ──┤                                              * ──> feature_out
                 └──> feature_phase ────────────────────────────┘
    """

    NORM_CLAMP = 2**-24  # Minimum positive subnormal for FP16

    def __init__(self, fiber: Fiber,
                 nonlinearity: torch.nn.Module = torch.nn.ReLU()):
        super().__init__()
        self.fiber = fiber
        self.nonlinearity = nonlinearity

        if len(set(fiber.channels)) == 1:
            # Fuse all the layer normalizations into a group normalization
            self.group_norm = torch.nn.GroupNorm(
                num_groups=len(fiber.degrees),
                num_channels=sum(fiber.channels))
        else:
            # Use multiple layer normalizations
            self.layer_norms = torch.nn.ModuleDict({
                str(degree): torch.nn.LayerNorm(channels)
                for degree, channels in fiber
            })

    def forward(self, features: Dict[str, Tensor], *args,
                **kwargs) -> Dict[str, Tensor]:
        output = {}
        if hasattr(self, "group_norm"):
            # Compute per-degree norms of features
            norms = [
                features[str(d)].norm(dim=-1,
                                      keepdim=True).clamp(min=self.NORM_CLAMP)
                for d in self.fiber.degrees
            ]
            fused_norms = torch.cat(norms, dim=-2)

            # Transform the norms only
            new_norms = self.nonlinearity(
                self.group_norm(fused_norms.squeeze(-1))).unsqueeze(-1)
            new_norms = torch.chunk(new_norms, chunks=len(self.fiber.degrees),
                                    dim=-2)

            # Scale features to the new norms
            for norm, new_norm, d in zip(norms, new_norms, self.fiber.degrees):
                output[str(d)] = features[str(d)] / norm * new_norm
        else:
            for degree, feat in features.items():
                norm = feat.norm(dim=-1,
                                 keepdim=True).clamp(min=self.NORM_CLAMP)
                new_norm = self.nonlinearity(self.layer_norms[degree](
                    norm.squeeze(-1)).unsqueeze(-1))
                output[degree] = new_norm * feat / norm

        return output
