import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, MessagePassing


class TemporalGCNConv(GCNConv):
    def __init__(self, in_channels, out_channels, num_time_steps, **kwargs):
        super(TemporalGCNConv, self).__init__(in_channels, out_channels, **kwargs)
        self.num_time_steps = num_time_steps

    def forward(self, x, edge_index, edge_weight=None, time_index=None):
        temporal_output = []

        for t in range(self.num_time_steps):
            # Process graph at time step t
            t_output = super(TemporalGCNConv, self).forward(x, edge_index, edge_weight)
            temporal_output.append(t_output)

        temporal_output = torch.stack(
            temporal_output, dim=0
        )  # Stack over time dimension
        return temporal_output


class TemporalGGNNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_time_steps):
        super(TemporalGGNNConv, self).__init__(aggr="add")
        self.num_time_steps = num_time_steps
        self.lin = nn.Linear(in_channels, out_channels)
        self.temporal_lin = nn.Linear(out_channels * num_time_steps, out_channels)

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


class TemporalTransformerConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_time_steps, heads=1):
        super(TemporalTransformerConv, self).__init__(aggr="add")
        self.num_time_steps = num_time_steps
        self.heads = heads

        self.lin = nn.Linear(in_channels, out_channels)
        self.temporal_lin = nn.Linear(out_channels * num_time_steps, out_channels)
        self.att_lin = nn.Linear(out_channels, heads * num_time_steps)

    def forward(self, x, edge_index, time_index):
        edge_index, _ = self.add_remaining_self_loops(edge_index)

        temporal_outputs = []
        for t in range(self.num_time_steps):
            t_edge_index = edge_index[:, time_index == t]
            t_x = self.propagate(t_edge_index, x=x)
            temporal_outputs.append(t_x)

        x = torch.cat(temporal_outputs, dim=-1)
        x = F.relu(self.temporal_lin(x))
        x = self.att_lin(x).view(-1, self.heads, self.num_time_steps)

        return x

    def message(self, x_j):
        return x_j


class TemporalGINConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_time_steps):
        super(TemporalGINConv, self).__init__(aggr="add")
        self.num_time_steps = num_time_steps
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )
        self.temporal_lin = nn.Linear(out_channels * num_time_steps, out_channels)

    def forward(self, x, edge_index, time_index):
        edge_index, _ = self.add_remaining_self_loops(edge_index)

        temporal_outputs = []
        for t in range(self.num_time_steps):
            t_edge_index = edge_index[:, time_index == t]
            t_x = self.propagate(t_edge_index, x=x)
            temporal_outputs.append(t_x)

        x = torch.cat(temporal_outputs, dim=-1)
        x = F.relu(self.temporal_lin(x))
        x = self.mlp(x)

        return x

    def message(self, x_j):
        return x_j


class TemporalGraphLSTMConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_time_steps):
        super(TemporalGraphLSTMConv, self).__init__(aggr="add")
        self.num_time_steps = num_time_steps
        self.lstm = nn.LSTMCell(in_channels, out_channels)
        self.temporal_lin = nn.Linear(out_channels * num_time_steps, out_channels)

    def forward(self, x, edge_index, time_index, h=None, c=None):
        edge_index, _ = self.add_remaining_self_loops(edge_index)

        temporal_outputs = []
        for t in range(self.num_time_steps):
            t_edge_index = edge_index[:, time_index == t]
            t_x = self.propagate(t_edge_index, x=x)
            h, c = self.lstm(t_x, (h, c))
            temporal_outputs.append(h)

        x = torch.cat(temporal_outputs, dim=-1)
        x = F.relu(self.temporal_lin(x))

        return x

    def message(self, x_j):
        return x


class TemporalDyEGNNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_time_steps):
        super(TemporalDyEGNNConv, self).__init__(aggr="add")
        self.num_time_steps = num_time_steps
        self.lin = nn.Linear(in_channels, out_channels)
        self.temporal_lin = nn.Linear(out_channels * num_time_steps, out_channels)
        self.att = nn.Sequential(nn.Linear(out_channels * 2, 1), nn.Sigmoid())

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

    def message(self, x_i, x_j):
        edge_features = torch.cat([x_i, x_j], dim=-1)
        attention_weights = self.att(edge_features)
        return x_j * attention_weights


class TemporalGRUConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_time_steps):
        super(TemporalGRUConv, self).__init__(aggr="add")
        self.num_time_steps = num_time_steps
        self.gru = nn.GRUCell(in_channels, out_channels)
        self.temporal_lin = nn.Linear(out_channels * num_time_steps, out_channels)

    def forward(self, x, edge_index, time_index, h=None):
        edge_index, _ = self.add_remaining_self_loops(edge_index)

        temporal_outputs = []
        for t in range(self.num_time_steps):
            t_edge_index = edge_index[:, time_index == t]
            t_x = self.propagate(t_edge_index, x=x)
            h = self.gru(t_x, h)
            temporal_outputs.append(h)

        x = torch.cat(temporal_outputs, dim=-1)
        x = F.relu(self.temporal_lin(x))

        return x

    def message(self, x_j):
        return x_j
