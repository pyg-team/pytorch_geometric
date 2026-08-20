from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn.aggr import Aggregation
from torch_geometric.utils import scatter, softmax


class SSMAAggregation(Aggregation):
    r"""Performs the Sequential Signal Mixing Aggregation (SSMA) method from
    `"Sequential Signal Mixing Aggregation for Message Passing Graph
    Neural Networks" <https://arxiv.org/abs/2409.19414>`_ paper.

    .. math::
            \mathrm{SSMA}(\mathcal{X}) = C* \mathcal{FFT}^{-1}
            \left( \prod_{\mathbf{x}_i \in \mathcal{X}}\mathcal{FFT}
            \left(A*\mathbf{x}_i+b \right) \right) + d


    Args:
        in_dim (int): The input dimension of the node features
        num_neighbors (int): Maximal number of neighbors to aggregate per node
        mlp_compression (float): The compression ratio for the last MLP,
                                if less than 1.0, the MLP will be factorized
                                (default: :obj:`1.0`)
        use_attention (bool): If True will use attention mechanism for
                              selecting the neighbors,
                              otherwise will use all neighbors.
                              (default: :obj:`True`)
        n_heads (int): Number of attention heads to use,
                       if "use_attention" is True. (default: :obj:`1`)
        temp (float): The attention temperature to use,
                      if "use_attention" is True. (default: :obj:`1.0`)
        att_feature (str): The feature to use for computing the attention
                           weights, if "use_attention" is True.
                           (default: :obj:`x`)
        learn_affine (bool): If True, will learn the affine transformation,
                             otherwise will use a fixed one.
                             (default: :obj:`False`)
    """
    def __init__(self, in_dim: int, num_neighbors: int,
                 mlp_compression: float = 1.0, use_attention: bool = True,
                 n_heads: int = 1, temp: float = 1.0, att_feature: str = "x",
                 learn_affine: bool = False):
        super().__init__()

        self._in_dim = in_dim
        self._max_neighbors = num_neighbors
        self._mlp_compression = mlp_compression
        self._n_heads = n_heads
        self._use_attention = use_attention
        self._attention_temp = temp
        self._att_feature = att_feature
        self._learn_affine = learn_affine

        att_groups = n_heads * num_neighbors

        if use_attention:
            self.attn_l = nn.LazyLinear(att_groups, bias=True)
            self.attn_r = nn.LazyLinear(att_groups, bias=True)
            self._neighbor_att_temp = temp
            self._edge_attention_ste = None

        m1 = self._max_neighbors + 1
        m2 = int((in_dim - 1) * self._max_neighbors + 1)
        self._m1 = m1
        self._m2 = m2

        # Set frozen affine layer
        self._affine_layer = nn.Linear(in_features=in_dim,
                                       out_features=self._m1 * self._m2,
                                       bias=True)
        aff_w = torch.zeros(in_dim, self._m1 * self._m2)
        aff_b = torch.zeros(self._m1 * self._m2, dtype=torch.float32)
        aff_w[:in_dim, :in_dim] = -torch.eye(in_dim, dtype=torch.float32)
        aff_b[self._m2] = 1
        self._affine_layer.weight.data = aff_w.T
        self._affine_layer.bias.data = aff_b
        if not learn_affine:
            for p in self._affine_layer.parameters():
                p.requires_grad = False

        if mlp_compression < 1.0:  # Perform matrix factorization
            T = (mlp_compression *
                 (self._m1 * self._m2 * in_dim)) / (self._m1 * self._m2 +
                                                    in_dim)
            T = int(np.ceil(T))
            self._mlp = nn.Sequential(
                nn.Linear(in_features=self._m1 * self._m2, out_features=T),
                nn.Linear(in_features=T, out_features=in_dim))
        else:
            self._mlp = nn.Linear(in_features=self._m1 * self._m2,
                                  out_features=in_dim, bias=True)
        self._pre_hook_run = False

    def _compute_attention(self, x: torch.Tensor,
                           edge_index: torch.Tensor) -> torch.Tensor:
        # Compute attention weights, each input is split into groups,
        # each group has its own attention per number of neighbors.
        # So we have  #groups * #neighbors attention weights
        x_l = self.attn_l(x.reshape(x.size(0), -1))
        x_r = self.attn_r(x.reshape(x.size(0), -1))

        x_l = x_l.reshape(x.size(0), -1,
                          self._max_neighbors)  # [N, #groups, #neighbors]
        x_r = x_r.reshape(x.size(0), -1,
                          self._max_neighbors)  # [N, #groups, #neighbors]

        # Compute softmax over the neighbors based on the attention weights
        # and the graph topology
        edge_attention_ste = softmax(
            F.leaky_relu(x_l[edge_index[0]] + x_r[edge_index[1]]) /
            self._neighbor_att_temp, edge_index[1])  # [E, #groups, #neighbors]
        return edge_attention_ste

    def pre_aggregation_hook(self, module, inputs):
        """Hook to run before the forward pass of the layer.
        This hook is needed only if we use attention mechanism in SSMA.
        The hook computes the attention weights given the source and target
        nodes.
        """
        if self._use_attention:
            edge_index, size, kwargs = inputs
            x = kwargs[self._att_feature]
            if isinstance(x, tuple):
                x = x[0]
            edge_attention_ste = (self._compute_attention(
                x=x, edge_index=edge_index.clone()))
            self._edge_attention_ste = edge_attention_ste
        self._pre_hook_run = True
        return None

    def forward(
        self,
        x: Tensor,
        index: Optional[Tensor] = None,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
        max_num_elements: Optional[int] = None,
    ) -> Tensor:
        assert (not self._use_attention) or self._pre_hook_run, \
            (
                "Have to run pre hook first when using attention, "
                "please make sure you register "
                "'pre_aggregation_hook' to the message passing layer: "
                "layer.register_propagate_forward_pre_hook("
                "ssma.pre_aggregation_hook)")

        x_in_shape = x.shape
        x = x.reshape(x.size(0), self._n_heads, -1)

        if self._use_attention:
            x_ = x.unsqueeze(2)  # [E, #groups, 1, #hidden]
            edge_att = self._edge_attention_ste.unsqueeze(
                -1)  # [E, #groups, #neighbors, 1]
            x_ = scatter(src=x_ * edge_att, index=index, dim=0,
                         reduce="sum")  # [E~, #groups, #neighbors, #hidden]
            x_ = torch.permute(x_, (2, 0, 1, 3))
            x = x_.reshape(x_.shape[0] * x_.shape[1], x_.shape[2],
                           x_.shape[3])  # [E~ * #neighbors, #groups, #hidden]
            index = torch.arange(dim_size,
                                 device=x.device).repeat(edge_att.size(2))

        # Perform affine transformation
        x_aff = self._affine_layer(x)

        # Compute FFT
        x_aff = x_aff.reshape(*x_aff.shape[:-1], self._m1, self._m2)
        x_fft = torch.fft.fft2(x_aff)

        # Aggregate neighbors
        x_fft_abs = x_fft.abs()
        x_fft_abs_log = (x_fft_abs + 1e-6).log()
        x_fft_angle = x_fft.angle()

        x_fft_abs_agg = scatter(src=x_fft_abs_log, index=index, dim=0,
                                dim_size=dim_size, reduce="mean").exp()
        x_fft_angle_agg = scatter(src=x_fft_angle, index=index, dim=0,
                                  dim_size=dim_size, reduce="sum")
        x_fft_agg = torch.polar(abs=x_fft_abs_agg, angle=x_fft_angle_agg)

        # Perform IFFT
        x_agg_comp = torch.fft.ifft2(x_fft_agg)
        x_agg = x_agg_comp.real

        # Perform MLP
        x_agg_transformed = self._mlp(x_agg.reshape(*x_agg.shape[:-2], -1))

        self._pre_hook_run = False

        x_agg_transformed = x_agg_transformed.reshape(-1, *x_in_shape[1:])
        return x_agg_transformed

    def __repr__(self) -> str:
        return "".join((f'SSMAAggregation(in_dim={self._in_dim},'
                        f'num_neighbors={self._max_neighbors},'
                        f'mlp_compression={self._mlp_compression},'
                        f'use_attention={self._use_attention},'
                        f'n_heads={self._n_heads},'
                        f'temp={self._attention_temp},'
                        f'att_feature=\'{self._att_feature}\',',
                        f'learn_affine={self._learn_affine})'))
