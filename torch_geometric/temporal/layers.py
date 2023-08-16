import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, MessagePassing


class TemporalGCNConv(GCNConv):
    def __init__(self, in_channels, out_channels, num_time_steps, **kwargs):
        super(TemporalGCNConv, self).__init__(in_channels, out_channels,
                                              **kwargs)
        self.num_time_steps = num_time_steps

    def forward(self, x, edge_index, edge_weight=None, time_index=None):
        temporal_output = []

        for t in range(self.num_time_steps):
            # Process graph at time step t
            t_output = super(TemporalGCNConv,
                             self).forward(x, edge_index, edge_weight)
            temporal_output.append(t_output)

        temporal_output = torch.stack(temporal_output,
                                      dim=0)  # Stack over time dimension
        return temporal_output


class TemporalGGNNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_time_steps):
        super(TemporalGGNNConv, self).__init__(aggr="add")
        self.num_time_steps = num_time_steps
        self.lin = nn.Linear(in_channels, out_channels)
        self.temporal_lin = nn.Linear(out_channels * num_time_steps,
                                      out_channels)

    def forward(self, x, edge_index, time_index):
        edge_index, _ = self.add_remaining_self_loops(edge_index)

        temporal_outputs = []
        for t in range(self.num_time_steps):
            t_edge_index = edge_index[:, time_index == t]
            t_x = self.propagate(t_edge_index, x=x)
            temporal_outputs.append(t_x)

        x = torch.cat(temporal_outputs, dim=-1)
        x = F.relu(self.temporal_lin(x))
        return x

    def message(self, x_j):
        return x_j
