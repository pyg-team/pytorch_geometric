import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import SAGEConv, Set2Set


class Set2SetNet(torch.nn.Module):
    r"""Model from the `"Order Matters: Sequence to sequence for sets"
    <https://arxiv.org/abs/1511.06391>`_ paper
    :class:`torch_geometric.nn.pool.DiffPool`.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Hidden node feature dimensionality.
        num_layers (int): Number of GNN layers.
    """

    def __init__(self, in_channels, hidden_channels, num_layers):
        super(Set2SetNet, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.set2set = Set2Set(hidden_channels, processing_steps=4)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.set2set.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = self.set2set(x, batch)
        return x

    def __repr__(self):
        return self.__class__.__name__