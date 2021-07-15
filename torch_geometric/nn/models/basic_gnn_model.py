import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import SAGEConv, global_mean_pool, JumpingKnowledge

class Basic_GNN_model_without_convs(torch.nn.Module):
    r"""Basic GNN model architecture that can be used as a template to inherit from
    for standard GNN models.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Hidden node feature dimensionality.
        num_layers (int): Number of Conv layers.
    """
    def __init__(self):
        return

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = global_mean_pool(x, batch)
        return x

    def __repr__(self):
        return self.__class__.__name__


class Basic_GNN_model(torch.nn.Module):
    r"""Basic GNN model architecture that can be used as a template to inherit from
    for standard GNN models.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Hidden node feature dimensionality.
        num_layers (int): Number of Conv layers.
    """
    def __init__(self, conv_layer, in_channels, hidden_channels, num_layers):
        super(Basic_GNN_model, self).__init__()
        self.conv1 = conv_layer(in_channels, hidden_channels)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(conv_layer(hidden_channels, hidden_channels))

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = global_mean_pool(x, batch)
        return x

    def __repr__(self):
        return self.__class__.__name__

class Basic_GNN_model_WithJK(torch.nn.Module):
    r"""Basic GNN model architecture that can be used as a template to inherit from
    for standard GNN models with the :class:`torch_geometric.nn.JumpingKnowledge` layer.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Hidden node feature dimensionality.
        num_layers (int): Number of GNN layers.
        mode (str): The aggregation mode for the :class:`torch_geometric.nn.conv.JumpingKnowledge` layer
            (default: :obj:`cat`). If mode = 'cat' the output feature has size num_layers * hidden_channels,
            otherwise the output feature has size hidden_channels.
    """
    def __init__(self, conv_layer, in_channels, hidden_channels, num_layers, mode='cat'):
        super(Basic_GNN_model_WithJK, self).__init__()
        self.conv1 = conv_layer(in_channels, hidden_channels)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(conv_layer(hidden_channels, hidden_channels))
        self.jump = JumpingKnowledge(mode)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.jump.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        xs = [x]
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            xs += [x]
        x = self.jump(xs)
        x = global_mean_pool(x, batch)
        return x

    def __repr__(self):
        return self.__class__.__name__