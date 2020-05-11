import torch
import torch.nn as nn
import numpy as np

from torch_geometric.nn import global_add_pool


class SchNet(nn.Module):
    def __init__(
            self,
            num_features,
            dim=128,
            n_filters=128,
            n_interactions=6,
            cutoff=10.0,
            n_gaussians=50,
    ):
        super(SchNet, self).__init__()
        self.lin = nn.Linear(num_features, dim, bias=False)
        self.distance_expansion = GaussianSmearing(0.0, cutoff, n_gaussians)
        self.interactions = nn.ModuleList([
            SchNetInteraction(dim, n_gaussians, n_filters, cutoff)
            for _ in range(n_interactions)
        ])
        self.out = nn.Sequential(nn.Linear(dim, dim // 2), ShiftedSoftplus(),
                                 nn.Linear(dim // 2, dim // 4),
                                 ShiftedSoftplus(), nn.Linear(dim // 4, 1))

    def forward(self, data):
        ind_i = data.dist_index[0]
        ind_j = data.dist_index[1]
        r_ij = data.dist
        x = data.x

        x = self.lin(x)

        f_ij = self.distance_expansion(r_ij)
        for interaction in self.interactions:
            v = interaction(x, r_ij, f_ij, ind_i, ind_j)
            x = x + v
        x = self.out(x)
        x = global_add_pool(x, data.batch)
        return x.view(-1)


class SchNetInteraction(nn.Module):
    def __init__(self, dim, n_spatial_basis, n_filters, cutoff):
        super(SchNetInteraction, self).__init__()
        self.filter_network = nn.Sequential(
            nn.Linear(n_spatial_basis, n_filters), ShiftedSoftplus(),
            nn.Linear(n_filters, n_filters))
        self.cutoff_network = CosineCutoff(cutoff)
        self.cfconv = CFConv(dim, n_filters, dim, self.filter_network,
                             self.cutoff_network)
        self.lin = nn.Linear(dim, dim)

    def forward(self, x, r_ij, f_ij, ind_i, ind_j):
        v = self.cfconv(x, r_ij, f_ij, ind_i, ind_j)
        v = self.lin(v)
        return v


class CosineCutoff(nn.Module):
    def __init__(self, cutoff):
        super(CosineCutoff, self).__init__()
        self.register_buffer("cutoff", torch.FloatTensor([cutoff]))

    def forward(self, distances):
        cutoffs = 0.5 * (torch.cos(distances * np.pi / self.cutoff) + 1.0)
        cutoffs *= (distances < self.cutoff).float()
        return cutoffs


class CFConv(nn.Module):
    def __init__(self, dim, n_filters, out_dim, filter_network,
                 cutoff_network):
        super(CFConv, self).__init__()
        self.in2f = nn.Linear(dim, n_filters, bias=False)
        self.f2out = nn.Sequential(nn.Linear(n_filters, out_dim),
                                   ShiftedSoftplus())
        self.filter_network = filter_network
        self.cutoff_network = cutoff_network

    def forward(self, x, r_ij, f_ij, ind_i, ind_j):
        W = self.filter_network(f_ij)
        C = self.cutoff_network(r_ij)
        W = W * C
        x = self.in2f(x)
        size = x.shape[0]
        x = x[ind_j] * W
        x = global_add_pool(x, ind_i, size=size)
        x = self.f2out(x)
        return x


class GaussianSmearing(nn.Module):
    def __init__(self, start=0.0, stop=5.0, n_gaussians=50):
        super(GaussianSmearing, self).__init__()
        offsets = torch.linspace(start, stop, n_gaussians)
        widths = torch.FloatTensor(
            (offsets[1] - offsets[0] * torch.ones_like(offsets)))
        self.register_buffer("widths", widths)
        self.register_buffer("offsets", offsets)

    def forward(self, distances):
        coeff = -0.5 / torch.pow(self.widths, 2)
        gauss = torch.exp(coeff * torch.pow(distances - self.offsets, 2))
        return gauss


class ShiftedSoftplus(nn.Module):
    def __init__(self, beta=1, threshhold=20, shift=None):
        super(ShiftedSoftplus, self).__init__()
        self.sp = nn.Softplus(beta, threshhold)
        if shift is None:
            shift = np.log(2.0)
        self.shift = shift

    def forward(self, x):
        return self.sp(x) - self.shift
