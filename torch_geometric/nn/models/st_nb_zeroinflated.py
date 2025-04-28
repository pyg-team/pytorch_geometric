import torch.nn as nn

from torch_geometric.nn.models.bidirectional_tcn import BTCN
from torch_geometric.nn.models.diffusion_gcn import DiffusionGCN
from torch_geometric.nn.models.nbnorm_zeroinflated import NBNormZeroInflated


class ST_NB_ZeroInflated(nn.Module):
    """Spatiotemporal Zero-Inflated Negative Binomial (ZINB) Model.

    This model combines spatial and temporal convolutions to process
    graph-structured data and learn spatiotemporal dynamics.
    The ZINB component models data with both overdispersion
    and zero-inflation.

    The architecture:
        wx_t  + wx_s
          |       |
         TC3     SC3
          |       |
         TC2     SC2
          |       |
         TC1     SC1
          |       |
         x_t     x_s

    Args:
        num_timesteps_input (int): Number of input time steps.
        hidden_dim_s (int): Hidden dimension size for spatial
        convolution layers.
        rank_s (int): Rank for spatial convolution latent representations.
        num_timesteps_output (int): Number of output time steps.
        space_dim (int): Spatial dimension (number of nodes in the graph).
        hidden_dim_t (int): Hidden dimension size for temporal
        convolution layers.
        rank_t (int): Rank for temporal convolution latent representations.

    Attributes:
        TC1, TC2, TC3 (BTCN): Temporal convolution layers.
        TNB (NBNormZeroInflated): Temporal ZINB module.
        SC1, SC2, SC3 (DiffusionGCN): Spatial convolution layers.
        SNB (NBNormZeroInflated): Spatial ZINB module.
    """
    def __init__(self, num_timesteps_input, hidden_dim_s, rank_s,
                 num_timesteps_output, space_dim, hidden_dim_t, rank_t):
        super().__init__()
        # Temporal convolution layers
        self.TC1 = BTCN(space_dim, hidden_dim_t)
        self.TC2 = BTCN(hidden_dim_t, rank_t)
        self.TC3 = BTCN(rank_t, hidden_dim_t)
        self.TNB = NBNormZeroInflated(hidden_dim_t, space_dim)

        # Spatial convolution layers
        self.SC1 = DiffusionGCN(num_timesteps_input, hidden_dim_s, 3)
        self.SC2 = DiffusionGCN(hidden_dim_s, rank_s, 2)
        self.SC3 = DiffusionGCN(rank_s, hidden_dim_s, 2)
        self.SNB = NBNormZeroInflated(hidden_dim_s, num_timesteps_output)

    def forward(self, X, A_q, A_h):
        """Forward pass of the ST_NB_ZeroInflated model.

        Args:
            X (torch.Tensor): Input data of shape (batch_size,
            num_timesteps, num_nodes).
            A_q (torch.Tensor): Forward random walk matrix (num_nodes,
            num_nodes).
            A_h (torch.Tensor): Backward random walk matrix (num_nodes,
            num_nodes).

        Returns:
            tuple: Tuple containing:
                - n_res (torch.Tensor): Count parameter of ZINB for the data.
                - p_res (torch.Tensor): Success probability parameter of ZINB.
                - pi_res (torch.Tensor): Zero-inflation parameter of ZINB.
        """
        # Process temporal features
        X_T = X.permute(0, 2, 1)  # (batch_size, num_nodes, num_timesteps)
        X_t1 = self.TC1(X_T)
        X_t2 = self.TC2(X_t1)  # Temporal factors
        self.temporal_factors = X_t2  # Cache for potential external use
        X_t3 = self.TC3(X_t2)
        _b, _h, _ht = X_t3.shape
        n_t_nb, p_t_nb, pi_t_nb = self.TNB(X_t3.view(_b, _h, _ht, 1))

        # Process spatial features
        X_s1 = self.SC1(X, A_q, A_h)
        X_s2 = self.SC2(X_s1, A_q, A_h)  # Spatial factors
        self.space_factors = X_s2  # Cache for potential external use
        X_s3 = self.SC3(X_s2, A_q, A_h)
        _b, _n, _hs = X_s3.shape
        n_s_nb, p_s_nb, pi_s_nb = self.SNB(X_s3.view(_b, _n, _hs, 1))

        # Combine spatial and temporal ZINB parameters
        n_res = n_t_nb.permute(0, 2, 1) * n_s_nb
        p_res = p_t_nb.permute(0, 2, 1) * p_s_nb
        pi_res = pi_t_nb.permute(0, 2, 1) * pi_s_nb

        return n_res, p_res, pi_res
