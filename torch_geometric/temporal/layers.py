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


class TemporalTransformerConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_time_steps, heads=1):
        super(TemporalTransformerConv, self).__init__(aggr="add")
        self.num_time_steps = num_time_steps
        self.heads = heads

        self.lin = nn.Linear(in_channels, out_channels)
        self.temporal_lin = nn.Linear(out_channels * num_time_steps,
                                      out_channels)
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
        x = self.mlp(x)

        return x

    def message(self, x_j):
        return x_j


class TemporalGraphLSTMConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_time_steps):
        super(TemporalGraphLSTMConv, self).__init__(aggr="add")
        self.num_time_steps = num_time_steps
        self.lstm = nn.LSTMCell(in_channels, out_channels)
        self.temporal_lin = nn.Linear(out_channels * num_time_steps,
                                      out_channels)

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
        return x_j


class TemporalDyEGNNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_time_steps):
        super(TemporalDyEGNNConv, self).__init__(aggr="add")
        self.num_time_steps = num_time_steps
        self.lin = nn.Linear(in_channels, out_channels)
        self.temporal_lin = nn.Linear(out_channels * num_time_steps,
                                      out_channels)
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
        self.temporal_lin = nn.Linear(out_channels * num_time_steps,
                                      out_channels)

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


class TemporalGSTTransformerConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_time_steps, num_heads=1):
        super(TemporalGSTTransformerConv, self).__init__(aggr="add")
        self.num_time_steps = num_time_steps
        self.num_heads = num_heads
        self.lin = nn.Linear(in_channels, out_channels)
        self.temporal_lin = nn.Linear(out_channels * num_time_steps,
                                      out_channels)
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=out_channels, nhead=num_heads,
            dim_feedforward=2 * out_channels)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=self.transformer_layer,
            num_layers=2,  # You can adjust the number of layers as needed
        )

    def forward(self, x, edge_index, time_index):
        edge_index, _ = self.add_remaining_self_loops(edge_index)

        temporal_outputs = []
        for t in range(self.num_time_steps):
            t_edge_index = edge_index[:, time_index == t]
            t_x = self.propagate(t_edge_index, x=x)
            temporal_outputs.append(t_x)

        x = torch.cat(temporal_outputs, dim=-1)
        x = F.relu(self.temporal_lin(x))

        # Apply spatial-temporal attention using Transformer
        x = x.permute(1, 0, 2)  # Transformer expects sequence length first
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # Reshape back to original shape

        return x

    def message(self, x_j):
        return x_j


class TemporalGSANConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_time_steps, heads=1):
        super(TemporalGSANConv, self).__init__(aggr="add")
        self.num_time_steps = num_time_steps
        self.heads = heads
        self.lin = nn.Linear(in_channels, out_channels)
        self.temporal_lin = nn.Linear(out_channels * num_time_steps,
                                      out_channels)
        self.att = nn.Sequential(
            nn.Linear(out_channels, heads * num_time_steps),
            nn.Softmax(dim=-1))

    def forward(self, x, edge_index, time_index):
        edge_index, _ = self.add_remaining_self_loops(edge_index)

        temporal_outputs = []
        for t in range(self.num_time_steps):
            t_edge_index = edge_index[:, time_index == t]
            t_x = self.propagate(t_edge_index, x=x)
            temporal_outputs.append(t_x)

        x = torch.cat(temporal_outputs, dim=-1)
        x = F.relu(self.temporal_lin(x))

        # Apply spatial attention across time steps
        x = x.view(x.size(0), self.heads, self.num_time_steps, -1)
        attention_weights = self.att(x).view(x.size(0), -1)
        x = x.view(x.size(0), self.heads * self.num_time_steps, -1)

        # Aggregate using attention weights
        x = self.aggregate(x, edge_index, attention_weights,
                           size=(x.size(0), x.size(0)))

        return x

    def message(self, x_j, attention_weights):
        return x_j * attention_weights.unsqueeze(-1)


class TemporalASConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_time_steps, num_layers):
        super(TemporalASConv, self).__init__(aggr="add")
        self.num_time_steps = num_time_steps
        self.num_layers = num_layers
        self.lin = nn.Linear(in_channels, out_channels)
        self.temporal_lin = nn.Linear(out_channels * num_time_steps,
                                      out_channels)
        self.conv_layers = nn.ModuleList(
            [nn.Linear(out_channels, out_channels) for _ in range(num_layers)])

    def forward(self, x, edge_index, time_index):
        edge_index, _ = self.add_remaining_self_loops(edge_index)

        temporal_outputs = []
        for t in range(self.num_time_steps):
            t_edge_index = edge_index[:, time_index == t]
            t_x = self.propagate(t_edge_index, x=x)
            temporal_outputs.append(t_x)

        x = torch.cat(temporal_outputs, dim=-1)
        x = F.relu(self.temporal_lin(x))

        # Apply adaptive skip connections across layers
        skip_x = x
        for layer in self.conv_layers:
            x = layer(x)
            x += skip_x  # Add skip connection
            skip_x = x  # Update skip connection

        return x

    def message(self, x_j):
        return x_j


class TemporalAPNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_time_steps):
        super(TemporalAPNConv, self).__init__(aggr="add")
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


class TemporalGraphSAINT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_time_steps):
        super(TemporalGraphSAINT, self).__init__()
        self.num_time_steps = num_time_steps
        self.conv1 = nn.Linear(in_channels, hidden_channels)
        self.conv2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, time_index):
        # Temporal mini-batch sampling using GraphSAINT
        temporal_sampler = TemporalGraphSAINTSampler(edge_index, time_index,
                                                     batch_size=64,
                                                     num_steps=2)
        temporal_loader = temporal_sampler.loader()

        for batch_size, n_id, adjs in temporal_loader:
            adjs = [adj.to(x.device) for adj in adjs]
            x_batch = x[n_id]

            x_batch = self.conv1(x_batch)
            x_batch = F.relu(x_batch)
            x_batch = self.conv2(x_batch)

            # Aggregate temporal batch results back to the original node order
            x[n_id] = x_batch

        return x
