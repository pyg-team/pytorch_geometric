#reference from https://github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/master/torch_geometric_temporal/nn/attention/mstgcn.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import ChebConv
from torch_geometric.transforms import LaplacianLambdaMax


class MSTGCNBlock(nn.Module):
    r"""An implementation of the Multi-Component Spatial-Temporal Graph
    Convolution block from this paper: `"Attention Based Spatial-Temporal
    Graph Convolutional Networks for Traffic Flow Forecasting."
    <https://ojs.aaai.org/index.php/AAAI/article/view/3881>`_

    Args:
        in_channels (int): Number of input features.
        K (int): Order of Chebyshev polynomials. Degree is K-1.
        nb_chev_filters (int): Number of Chebyshev filters.
        nb_time_filters (int): Number of time filters.
        time_strides (int): Time strides during temporal convolution.
    """
    def __init__(
        self,
        in_channels: int,
        K: int,
        nb_chev_filter: int,
        nb_time_filter: int,
        time_strides: int,
    ):
        super(MSTGCNBlock, self).__init__()

        self._cheb_conv = ChebConv(in_channels, nb_chev_filter, K,
                                   normalization=None)

        self._time_conv = nn.Conv2d(
            nb_chev_filter,
            nb_time_filter,
            kernel_size=(1, 3),
            stride=(1, time_strides),
            padding=(0, 1),
        )

        self._residual_conv = nn.Conv2d(in_channels, nb_time_filter,
                                        kernel_size=(1, 1),
                                        stride=(1, time_strides))

        self._layer_norm = nn.LayerNorm(nb_time_filter)
        self.nb_time_filter = nb_time_filter

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, X: torch.FloatTensor,
                edge_index: torch.LongTensor) -> torch.FloatTensor:
        """
        Making a forward pass with a single MSTGCN block.

        Arg types:
            * X (PyTorch FloatTensor) - Node features for T time periods, with shape (B, N_nodes, F_in, T_in).
            * edge_index (PyTorch LongTensor): Edge indices, can be an array of a list of Tensor arrays, depending on whether edges change over time.

        Return types:
            * X (PyTorch FloatTensor) - Hidden state tensor for all nodes, with shape (B, N_nodes, nb_time_filter, T_out).
        """

        batch_size, num_of_vertices, in_channels, num_of_timesteps = X.shape

        if not isinstance(edge_index, list):

            lambda_max = LaplacianLambdaMax()(
                Data(edge_index=edge_index, edge_attr=None,
                     num_nodes=num_of_vertices)).lambda_max

            X_tilde = X.permute(2, 0, 1, 3)
            X_tilde = X_tilde.reshape(num_of_vertices, in_channels,
                                      num_of_timesteps * batch_size)
            X_tilde = X_tilde.permute(2, 0, 1)
            X_tilde = F.relu(
                self._cheb_conv(x=X_tilde, edge_index=edge_index,
                                lambda_max=lambda_max))
            X_tilde = X_tilde.permute(1, 2, 0)
            X_tilde = X_tilde.reshape(num_of_vertices, self.nb_time_filter,
                                      batch_size, num_of_timesteps)
            X_tilde = X_tilde.permute(2, 0, 1, 3)

        else:
            X_tilde = []
            for t in range(num_of_timesteps):
                lambda_max = LaplacianLambdaMax()(Data(
                    edge_index=edge_index[t],
                    edge_attr=None,
                    num_nodes=num_of_vertices,
                )).lambda_max
                X_tilde.append(
                    torch.unsqueeze(
                        self._cheb_conv(X[:, :, :, t], edge_index[t],
                                        lambda_max=lambda_max),
                        -1,
                    ))
            X_tilde = F.relu(torch.cat(X_tilde, dim=-1))

        X_tilde = self._time_conv(X_tilde.permute(0, 2, 1, 3))
        X = self._residual_conv(X.permute(0, 2, 1, 3))
        X = self._layer_norm(F.relu(X + X_tilde).permute(0, 3, 2, 1))
        X = X.permute(0, 2, 3, 1)
        return X


class MSTGCN(nn.Module):
    r"""An implementation of the Multi-Component Spatial-Temporal Graph Convolution Networks, a degraded version of ASTGCN.
    For details see this paper: `"Attention Based Spatial-Temporal Graph Convolutional
    Networks for Traffic Flow Forecasting." <https://ojs.aaai.org/index.php/AAAI/article/view/3881>`_

    Args:

        nb_block (int): Number of ASTGCN blocks in the model.
        in_channels (int): Number of input features.
        K (int): Order of Chebyshev polynomials. Degree is K-1.
        nb_chev_filter (int): Number of Chebyshev filters.
        nb_time_filter (int): Number of time filters.
        time_strides (int): Time strides during temporal convolution.
        num_for_predict (int): Number of predictions to make in the future.
        len_input (int): Length of the input sequence.
    """
    def __init__(
        self,
        nb_block: int,
        in_channels: int,
        K: int,
        nb_chev_filter: int,
        nb_time_filter: int,
        time_strides: int,
        num_for_predict: int,
        len_input: int,
    ):
        super(MSTGCN, self).__init__()

        self._blocklist = nn.ModuleList([
            MSTGCNBlock(in_channels, K, nb_chev_filter, nb_time_filter,
                        time_strides)
        ])

        self._blocklist.extend([
            MSTGCNBlock(nb_time_filter, K, nb_chev_filter, nb_time_filter, 1)
            for _ in range(nb_block - 1)
        ])

        self._final_conv = nn.Conv2d(
            int(len_input / time_strides),
            num_for_predict,
            kernel_size=(1, nb_time_filter),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        """
        Resetting the model parameters.
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, X: torch.FloatTensor,
                edge_index: torch.LongTensor) -> torch.FloatTensor:
        r"""Making a forward pass. This module takes a likst of MSTGCN blocks and use a final convolution to serve as a multi-component fusion.
        B is the batch size. N_nodes is the number of nodes in the graph. F_in is the dimension of input features.
        T_in is the length of input sequence in time. T_out is the length of output sequence in time.

        Arg types:
            * X (PyTorch FloatTensor) - Node features for T time periods, with shape (B, N_nodes, F_in, T_in).
            * edge_index (PyTorch LongTensor): Edge indices, can be an array of a list of Tensor arrays, depending on whether edges change over time.

        Return types:
            * X (PyTorch FloatTensor) - Hidden state tensor for all nodes, with shape (B, N_nodes, T_out).
        """
        for block in self._blocklist:
            X = block(X, edge_index)

        X = self._final_conv(X.permute(0, 3, 1, 2))[:, :, :,
                                                    -1].permute(0, 2, 1)
        return X
