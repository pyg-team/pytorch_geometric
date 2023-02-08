from typing import Optional
import copy
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_sparse import SparseTensor, mul
from torch_sparse import sum as sparsesum

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj, OptPairTensor, OptTensor
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.testing import is_full_test
# for the last version of torch_geometric
from torch_geometric.utils import scatter, spmm
from torch_geometric.utils.num_nodes import maybe_num_nodes

# Inspired from IntelLabs :
# https://github.com/IntelLabs/bayesian-torch/blob/main/bayesian_torch/
# layers/base_variational_layer.py


class BayesianGCNConv(MessagePassing):
    _cached_edge_index: Optional[OptPairTensor]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, in_channels: int, out_channels: int, prior_mean=0,
                 prior_variance=1, posterior_mu_init=0,
                 posterior_rho_init=-3.0, cached: bool = False,
                 normalize: bool = False, bias: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
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

        self.register_buffer('eps_weight',
                             torch.Tensor(out_channels, in_channels),
                             persistent=False)
        self.register_buffer('prior_weight_mu',
                             torch.Tensor(out_channels, in_channels),
                             persistent=False)
        self.register_buffer('prior_weight_sigma',
                             torch.Tensor(out_channels, in_channels),
                             persistent=False)

        if bias:
            self.mu_bias = Parameter(torch.Tensor(out_channels))
            self.rho_bias = Parameter(torch.Tensor(out_channels))
            self.register_buffer('eps_bias', torch.Tensor(out_channels),
                                 persistent=False)
            self.register_buffer('prior_bias_mu', torch.Tensor(out_channels),
                                 persistent=False)
            self.register_buffer('prior_bias_sigma',
                                 torch.Tensor(out_channels), persistent=False)
        else:
            self.register_buffer('prior_bias_mu', None, persistent=False)
            self.register_buffer('prior_bias_sigma', None, persistent=False)
            self.register_parameter('mu_bias', None)
            self.register_parameter('rho_bias', None)
            self.register_buffer('eps_bias', None, persistent=False)

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

    def kl_div(self, mu_q, sigma_q, mu_p, sigma_p):
        """
        Calculates kl divergence between two gaussians (Q || P)
        Parameters:
             * mu_q: torch.Tensor -> mu parameter of distribution Q
             * sigma_q: torch.Tensor -> sigma parameter of distribution Q
             * mu_p: float -> mu parameter of distribution P
             * sigma_p: float -> sigma parameter of distribution P
        returns torch.Tensor of shape 0
        """
        kl = torch.log(sigma_p) - torch.log(
            sigma_q) + (sigma_q**2 + (mu_q - mu_p)**2) / (2 *
                                                          (sigma_p**2)) - 0.5
        return kl.mean()

    def kl_loss(self):
        sigma_weight = torch.log1p(torch.exp(self.rho_weight))
        kl = self.kl_div(self.mu_weight, sigma_weight, self.prior_weight_mu,
                         self.prior_weight_sigma)
        if self.mu_bias is not None:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            kl += self.kl_div(self.mu_bias, sigma_bias, self.prior_bias_mu,
                              self.prior_bias_sigma)
        return kl

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None,
                return_kl: bool = True) -> Tensor:
        """"""

        sigma_weight = torch.log1p(torch.exp(self.rho_weight))

        weight = self.mu_weight + \
            (sigma_weight * self.eps_weight.data.normal_())

        if return_kl:
            kl_weight = self.kl_div(self.mu_weight, sigma_weight,
                                    self.prior_weight_mu,
                                    self.prior_weight_sigma)

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
        weight = self.mu_weight + \
            (sigma_weight * self.eps_weight.data.normal_())
        if return_kl:
            kl_weight = self.kl_div(self.mu_weight, sigma_weight,
                                    self.prior_weight_mu,
                                    self.prior_weight_sigma)

        bias = None  # Initialization of bias

        if self.mu_bias is not None:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            bias = self.mu_bias + (sigma_bias * self.eps_bias.data.normal_())
            if return_kl:
                kl_bias = self.kl_div(self.mu_bias, sigma_bias,
                                      self.prior_bias_mu,
                                      self.prior_bias_sigma)

        x = F.linear(x, weight, bias)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)

        if return_kl:
            if self.mu_bias is not None:
                kl = kl_weight + kl_bias
            else:
                kl = kl_weight

            return out, kl

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)


def test_gcn_conv_with_sparse_input_feature(device="cpu"):
    x = torch.sparse_coo_tensor(indices=torch.tensor([[0, 0], [0, 1]]),
                                    values=torch.tensor([1., 1.]),
                                    size=torch.Size([4, 16])).to(device)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]]).to(device)
    conv = BayesianGCNConv(16, 32).to(device)
    out, kl = conv(x, edge_index)
    print(out.shape, kl.detach().cpu())
    assert out.size() == (4, 32)

def test_static_gcn_conv(device="cpu"):
    x = torch.randn(3, 4, 16).to(device)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]]).to(device)
    conv = BayesianGCNConv(16, 32).to(device)
    out, kl = conv(x, edge_index)
    print(out.shape, kl.detach().cpu())
    assert out.size() == (3,4,32)
    

def test_gcn_conv_with_decomposed_layers(device="cpu"):
    x = torch.randn(4, 16).to(device)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]]).to(device)
    conv = BayesianGCNConv(16, 32).to(device)
    decomposed_conv = copy.deepcopy(conv)
    decomposed_conv.decomposed_layers = 2
    out1, kl = conv(x, edge_index)
    out2, kl_2 = decomposed_conv(x, edge_index)
    print(out1.shape, out2.shape)
    # the weight differs from out1 to out2 because different weight and biases
    assert torch.allclose(out1, out2, atol=1.0)

    if is_full_test():
        t = '(Tensor, Tensor, OptTensor) -> Tensor'
        jit = torch.jit.script(decomposed_conv.jittable(t))
        assert jit(x, edge_index).tolist() == out1.tolist()

def test_gcn_conv(device="cpu"):
    x = torch.randn(4, 16).to(device)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]]).to(device)
    row, col = edge_index
    value = torch.rand(row.size(0)).to(device)
    zeros_values = torch.zeros(value.size())
    adj1 = torch.sparse_coo_tensor(edge_index, value, device=x.device)
    adj2 = torch.sparse_coo_tensor(edge_index, value, device=x.device)
    conv = BayesianGCNConv(16, 32).to(device)
    assert conv.__repr__() == 'BayesianGCNConv(16, 32)'
    out1, kl = conv(x, edge_index)
    print(out1.shape, kl.detach().cpu())
    assert out1.size() == (4, 32)

    out2, kl = conv(x, edge_index, value)
    print(out2.shape, kl.detach().cpu())
    assert out2.size() == (4, 32)

    if is_full_test():
        t = '(Tensor, Tensor, OptTensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert jit(x, edge_index).tolist() == out1.tolist()
        assert jit(x, edge_index, value).tolist() == out2.tolist()

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Testing simple Bayesian gcn_conv')
    test_gcn_conv(device)
    print('Testing Bayesian gcn_conv with sparse input feature')
    test_gcn_conv_with_sparse_input_feature(device)
    print('Static Test')
    test_static_gcn_conv(device)
    print('Test with decomposed layers')
    test_gcn_conv_with_decomposed_layers(device)
