import torch
import torch.nn.functional as F
from torch.nn import Linear, Conv1d
from torch_geometric.nn import SAGEConv, global_sort_pool


class SortPool(torch.nn.Module):
    r"""Model from  the `"An End-to-End Deep Learning
    Architecture for Graph Classification"
    <https://www.cse.wustl.edu/~muhan/papers/AAAI_2018_DGCNN.pdf>`_ paper making use of
    pooling operator :class:`torch_geometric.nn.pool.GlobalSortPool`.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Hidden node feature dimensionality.
        num_layers (int): Number of GNN layers.
    """
    def __init__(self,  in_channels, hidden_channels, num_layers):
        super(SortPool, self).__init__()
        self.k = 30
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.conv1d = Conv1d(hidden_channels, 32, 5)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.conv1d.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = global_sort_pool(x, batch, self.k)
        return x

    def __repr__(self):
        return self.__class__.__name__