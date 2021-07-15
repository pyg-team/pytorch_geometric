import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import SAGEConv, GlobalAttention


class GlobalAttentionNet(torch.nn.Module):
    r"""A graph model combining the use of
    :class:`torch_geometric.nn.GlobalAttention` attention layer and
    :class:`torch_geometric.nn.SAGEConv` convolutional layers.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Hidden node feature dimensionality.
        num_layers (int): Number of SAGEConv layers.
    """
    def __init__(self, in_channels, hidden_channels, num_layers):
        super(GlobalAttentionNet, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.att = GlobalAttention(Linear(hidden_channels, 1))

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.att.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = self.att(x, batch)
        return x

    def __repr__(self):
        return self.__class__.__name__