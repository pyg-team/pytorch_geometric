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


class HyperbolicGNNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_time_steps, c):
        super(HyperbolicGNNConv, self).__init__(aggr='add')
        self.num_time_steps = num_time_steps
        self.c = c
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

    def aggregate(self, inputs, index, dim_size=None):
        # Apply hyperbolic averaging for aggregation
        return hyperbolic_mean(inputs, index, c=self.c, dim=0, size=dim_size)


class TemporalHyperbolicGNN(nn.Module):
    def __init__(self, num_nodes, num_features, num_classes, num_time_steps,
                 c):
        super(TemporalHyperbolicGNN, self).__init__()
        self.conv1 = HyperbolicGNNConv(num_features, 64, num_time_steps, c)
        self.conv2 = HyperbolicGNNConv(64, num_classes, num_time_steps, c)

    def forward(self, x, edge_index, time_index):
        x = self.conv1(x, edge_index, time_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index, time_index)
        return x


def hyperbolic_mean(inputs, index, c, dim=0, size=None):
    num_nodes = size[0] if size is not None else index.max().item() + 1
    sum = torch_scatter.scatter_add(inputs, index, dim=dim, dim_size=num_nodes)
    count = torch_scatter.scatter_add(torch.ones_like(inputs), index, dim=dim,
                                      dim_size=num_nodes)

    mean = sum / count.clamp(min=1)
    norm_mean = (1 - c**2 * mean.norm(dim=-1)**2).unsqueeze(-1)

    return mean / norm_mean


class TemporalAGNNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_time_steps):
        super(TemporalAGNNConv, self).__init__(aggr='add')
        self.num_time_steps = num_time_steps
        self.lin = nn.Linear(in_channels, out_channels)
        self.temporal_lin = nn.Linear(out_channels * num_time_steps,
                                      out_channels)

    def forward(self, x, edge_index, time_index,
                adversarial_perturbation=None):
        edge_index, _ = self.add_remaining_self_loops(edge_index)

        temporal_outputs = []
        for t in range(self.num_time_steps):
            t_edge_index = edge_index[:, time_index == t]
            t_x = self.propagate(t_edge_index, x=x)
            temporal_outputs.append(t_x)

        x = torch.cat(temporal_outputs, dim=-1)
        x = F.relu(self.temporal_lin(x))

        if adversarial_perturbation is not None:
            x = x + adversarial_perturbation

        return x

    def message(self, x_j):
        return x_j


class TemporalAdversarialGNN(nn.Module):
    def __init__(self, num_nodes, num_features, num_classes, num_time_steps):
        super(TemporalAdversarialGNN, self).__init__()
        self.conv1 = TemporalAGNNConv(num_features, 64, num_time_steps)
        self.conv2 = TemporalAGNNConv(64, num_classes, num_time_steps)
        self.adversary = nn.Sequential(nn.Linear(num_classes, 128), nn.ReLU(),
                                       nn.Linear(128, num_features))

    def forward(self, x, edge_index, time_index):
        x = self.conv1(x, edge_index, time_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index, time_index)

        # Adversarial training
        adv_perturbation = self.adversary(x)
        x = self.conv1(x, edge_index, time_index,
                       adversarial_perturbation=adv_perturbation)

        return x


class TemporalGraphWaveNetConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_time_steps, dilation,
                 kernel_size):
        super(TemporalGraphWaveNetConv, self).__init__(aggr='add')
        self.num_time_steps = num_time_steps
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.temporal_conv = nn.Conv1d(out_channels * num_time_steps,
                                       out_channels, 1)

    def forward(self, x, edge_index, time_index):
        edge_index, _ = self.add_remaining_self_loops(edge_index)

        temporal_outputs = []
        for t in range(self.num_time_steps):
            t_edge_index = edge_index[:, time_index == t]
            t_x = self.propagate(t_edge_index, x=x)
            temporal_outputs.append(t_x)

        x = torch.cat(temporal_outputs, dim=-1)
        x = self.conv(x.unsqueeze(0)).squeeze(0)
        x = F.relu(x)

        # Apply temporal convolution
        x = self.temporal_conv(x.unsqueeze(0)).squeeze(0)

        return x

    def message(self, x_j):
        return x_j


class TemporalGraphWaveNet(nn.Module):
    def __init__(self, num_nodes, num_features, num_classes, num_time_steps):
        super(TemporalGraphWaveNet, self).__init__()
        self.conv1 = TemporalGraphWaveNetConv(num_features, 64, num_time_steps,
                                              dilation=2, kernel_size=3)
        self.conv2 = TemporalGraphWaveNetConv(64, num_classes, num_time_steps,
                                              dilation=4, kernel_size=3)

    def forward(self, x, edge_index, time_index):
        x = self.conv1(x, edge_index, time_index)
        x = self.conv2(x, edge_index, time_index)
        return x


class TemporalSSTGNNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_time_steps):
        super(TemporalSSTGNNConv, self).__init__(aggr='add')
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


class TemporalSSTGNN(nn.Module):
    def __init__(self, num_nodes, num_features, num_classes, num_time_steps):
        super(TemporalSSTGNN, self).__init__()
        self.conv1 = TemporalSSTGNNConv(num_features, 64, num_time_steps)
        self.conv2 = TemporalSSTGNNConv(64, num_classes, num_time_steps)
        self.self_supervised_linear = nn.Linear(num_classes, num_features)

    def forward(self, x, edge_index, time_index):
        x = self.conv1(x, edge_index, time_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index, time_index)

        # Self-supervised learning: Predict node features
        self_supervised_target = self.self_supervised_linear(x)

        return x, self_supervised_target
