from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, SparseTensor
from torch_geometric.utils import spmm


class BayesianGCNConv(MessagePassing):
    r"""The Bayesian graph convolutional operator
    inspired from the `"Improving model calibration with
    accuracy versus uncertainty optimization"
    <https://arxiv.org/abs/2012.07923>`_ paper

    We first proceed wth linear transformation:

    .. math::
        \mathbf{X}^{\prime} = \mathbf{X} \mathbf{W}^{T} +  \mathbf{b}

    where :math:`\mathbf{W}^{T}` denotes the weights with a
    fully factorized Gaussian distributions represented by :math:`\mu` and
    :math:`\sigma`, with an unconstrained parameter :math:`\rho`
    for :math:`\sigma = log(1+exp(\rho))` to avoid non-negative variance.
    :math:`\mathbf{b}` denotes the bias which is based on the same structure
    as the weights.
    CAUTION: the weights and :obj:`edge_weight` are different parameters,
    since the :obj:`edge_weight` are initialized to one,
    they are not considered as a Parameter.
    NOTE: the weights and bias are initialized with a Gaussian distribution
    of mean equal to :obj:`0.0` and a standard deviation of :obj:`1.0`.

    Then propagate the message through the graph with a graph convolution
    operator from the `"Semi-supervised Classification with Graph
    Convolutional Networks" <https://arxiv.org/abs/1609.02907>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.
    The adjacency matrix can include other values than :obj:`1` representing
    edge weights via the optional :obj:`edge_weight` tensor.

    Its node-wise formulation is given by:

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta}^{\top} \sum_{j \in
        \mathcal{N}(i) \cup \{ i \}} \frac{e_{j,i}}{\sqrt{\hat{d}_j
        \hat{d}_i}} \mathbf{x}_j

    with :math:`\hat{d}_i = 1 + \sum_{j \in \mathcal{N}(i)} e_{j,i}`, where
    :math:`e_{j,i}` denotes the edge weight from source node :obj:`j` to target
    node :obj:`i` (default: :obj:`1.0`)

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        out_channels (int): Size of each output sample.
        prior_mean (float): Mean of the prior arbitrary distribution to be
            used on the complexity cost (default: :obj:`0.0`)
        prior_variance (float): Variance of the prior arbitrary distribution
            to be used on the complexity cost (default: :obj:`1.0`)
        posterior_mu_init (float): Init trainable mu parameter representing
            mean of the approximate posterior (default: :obj:`-3.0`)
        posterior_rho_init (float): Init trainable rho parameter representing
            the sigma of the approximate posterior through softplus function
            (default: :obj:`0.0`)
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        normalize (bool, optional): Whether to add self-loops and compute
            symmetric normalization coefficients on the fly.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
         - **input:**
           node features :math:`(|\mathcal{V}|, F_{in})`,
           edge indices :math:`(2, |\mathcal{E}|)`,
           edge weights :math:`(|\mathcal{E}|)` *(optional)*
         - **output:**
           node features :math:`(|\mathcal{V}|, F_{out})`,
           Kullback-Leibler divergence for weights and bias
           :math:`(\mathrm{KL}[q_{\theta}(w)||p(w)])`
           with variational parameters :math:`(\theta = (\mu, \rho))`
           to make an approximation of the variational posterior
           :math:`(q(\theta)=\mathcal{N}(\mu, log(1+e^{\rho})))` *(optional)*
    """
    _cached_edge_index: Optional[OptPairTensor]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prior_mean: float = 0.0,
        prior_variance: float = 1.0,
        posterior_mu_init: float = 0.0,
        posterior_rho_init: float = -3.0,
        cached: bool = False,
        normalize: bool = False,
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init
        self.cached = cached
        self.normalize = normalize

        self.mu_weight = Parameter(torch.Tensor(out_channels, in_channels))
        self.rho_weight = Parameter(torch.Tensor(out_channels, in_channels))

        self.register_buffer("eps_weight",
                             torch.Tensor(out_channels, in_channels),
                             persistent=False)
        self.register_buffer("prior_weight_mu",
                             torch.Tensor(out_channels, in_channels),
                             persistent=False)
        self.register_buffer(
            "prior_weight_sigma",
            torch.Tensor(out_channels, in_channels),
            persistent=False,
        )

        if bias:
            self.mu_bias = Parameter(torch.Tensor(out_channels))
            self.rho_bias = Parameter(torch.Tensor(out_channels))
            self.register_buffer("eps_bias", torch.Tensor(out_channels),
                                 persistent=False)
            self.register_buffer("prior_bias_mu", torch.Tensor(out_channels),
                                 persistent=False)
            self.register_buffer("prior_bias_sigma",
                                 torch.Tensor(out_channels), persistent=False)
        else:
            self.register_buffer("prior_bias_mu", None, persistent=False)
            self.register_buffer("prior_bias_sigma", None, persistent=False)
            self.register_parameter("mu_bias", None)
            self.register_parameter("rho_bias", None)
            self.register_buffer("eps_bias", None, persistent=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.prior_weight_mu.fill_(self.prior_mean)
        self.prior_weight_sigma.fill_(self.prior_variance)

        self.mu_weight.data.normal_(mean=self.posterior_mu_init, std=0.1)
        self.rho_weight.data.normal_(mean=self.posterior_rho_init, std=0.1)
        if self.mu_bias is not None:
            self.prior_bias_mu.fill_(self.prior_mean)
            self.prior_bias_sigma.fill_(self.prior_variance)
            self.mu_bias.data.normal_(mean=self.posterior_mu_init, std=0.1)
            self.rho_bias.data.normal_(mean=self.posterior_rho_init, std=0.1)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def kl_div(
        self,
        mu_q: Tensor,
        sigma_q: Tensor,
        mu_p: Tensor,
        sigma_p: Tensor,
    ) -> Tensor:
        r"""
        Calculates kl divergence between two gaussians (:math:`Q || P`).

        Args:
            mu_q (Tensor): :math:`\mu` parameter of distribution :math:`Q`.
            sigma_q (Tensor): :math:`\sigma` parameter of
                distribution :math:`Q`.
            mu_p (Tensor): :math:`\mu` parameter of distribution :math:`P`.
            sigma_p (Tensor): :math:`\sigma` parameter of
                distribution :math:`P`.
        """
        kl = (torch.log(sigma_p) - torch.log(sigma_q) +
              (sigma_q**2 + (mu_q - mu_p)**2) / (2 * (sigma_p**2)) - 0.5)
        return kl.mean()

    def kl_loss(self) -> Tensor:
        sigma_weight = torch.log1p(torch.exp(self.rho_weight))
        kl = self.kl_div(self.mu_weight, sigma_weight, self.prior_weight_mu,
                         self.prior_weight_sigma)
        if self.mu_bias is not None:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            kl += self.kl_div(self.mu_bias, sigma_bias, self.prior_bias_mu,
                              self.prior_bias_sigma)
        return kl

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        return_kl_divergence: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """"""

        sigma_weight = torch.log1p(torch.exp(self.rho_weight))

        weight = self.mu_weight + (sigma_weight *
                                   self.eps_weight.data.normal_())

        kl_weight = self.kl_div(self.mu_weight, sigma_weight,
                                self.prior_weight_mu, self.prior_weight_sigma)

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

        elif isinstance(edge_index, SparseTensor):
            cache = self._cached_adj_t
            if cache is None:
                edge_index = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, x.size(self.node_dim), x.dtype)
                if self.cached:
                    self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        sigma_weight = torch.log1p(torch.exp(self.rho_weight))
        weight = self.mu_weight + (sigma_weight *
                                   self.eps_weight.data.normal_())

        kl_weight = self.kl_div(self.mu_weight, sigma_weight,
                                self.prior_weight_mu, self.prior_weight_sigma)

        bias = None

        if self.mu_bias is not None:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            bias = self.mu_bias + (sigma_bias * self.eps_bias.data.normal_())

            kl_bias = self.kl_div(self.mu_bias, sigma_bias, self.prior_bias_mu,
                                  self.prior_bias_sigma)

        x = F.linear(x, weight, bias)

        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)

        if self.mu_bias is not None:
            kl = kl_weight + kl_bias
        else:
            kl = kl_weight

        if return_kl_divergence:
            return out, kl
        else:
            return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)
