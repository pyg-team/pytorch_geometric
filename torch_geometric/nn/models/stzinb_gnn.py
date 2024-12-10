import torch
import torch.nn as nn
from torch.nn.functional import softplus

from torch_geometric.nn import GCNConv


class STZINBGNN(nn.Module):
    def __init__(self, num_nodes, num_features, time_window, hidden_dim_s,
                 hidden_dim_t, rank_s, rank_t, k):
        super().__init__()

        # Spatial Layers (Replaces D_GCN)
        self.spatial_conv1 = GCNConv(num_features, hidden_dim_s)
        self.spatial_conv2 = GCNConv(hidden_dim_s, rank_s)
        self.spatial_conv3 = GCNConv(rank_s, hidden_dim_s)

        # Temporal Layers (Replaces B_TCN)
        self.temporal_conv1 = nn.Conv1d(in_channels=hidden_dim_s,
                                        out_channels=hidden_dim_t,
                                        kernel_size=3, padding=1)
        self.temporal_conv2 = nn.Conv1d(in_channels=hidden_dim_t,
                                        out_channels=rank_t, kernel_size=3,
                                        padding=1)
        self.temporal_conv3 = nn.Conv1d(in_channels=rank_t,
                                        out_channels=hidden_dim_t,
                                        kernel_size=3, padding=1)

        # ZINB Layer Parameters
        self.fc_pi = nn.Linear(hidden_dim_t, 1)  # Zero-inflation parameter
        self.fc_n = nn.Linear(hidden_dim_t, 1)  # Shape parameter
        self.fc_p = nn.Linear(hidden_dim_t, 1)  # Probability parameter

        # Time windows for prediction
        self.k = k

    def forward(self, x, edge_index, batch):
        # Spatial Embedding
        x_s = torch.relu(self.spatial_conv1(x, edge_index))
        x_s = torch.relu(self.spatial_conv2(x_s, edge_index))
        x_s = torch.relu(self.spatial_conv3(x_s, edge_index))

        # Reshape for Temporal Processing
        num_graphs = batch.max().item() + 1
        x_s = x_s.view(num_graphs, -1,
                       x_s.size(-1))  # [num_graphs, num_nodes, hidden_dim_s]
        x_s = x_s.permute(0, 2,
                          1)  # For TCN: [batch_size, hidden_dim_s, num_nodes]

        # Temporal Embedding
        x_t = torch.relu(self.temporal_conv1(x_s))
        x_t = torch.relu(self.temporal_conv2(x_t))
        x_t = torch.relu(self.temporal_conv3(x_t))

        # Flatten for ZINB parameterization
        x_t = x_t.permute(0, 2, 1).reshape(
            -1, x_t.size(1))  # Combine spatial and temporal features

        # ZINB Parameters
        pi = torch.sigmoid(
            self.fc_pi(x_t))  # Shape: [batch_size * num_nodes, 1]
        n = softplus(self.fc_n(x_t))  # Shape: [batch_size * num_nodes, 1]
        p = torch.sigmoid(self.fc_p(x_t))  # Shape: [batch_size * num_nodes, 1]

        return pi, n, p
