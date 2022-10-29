import torch
import torch.nn.functional as F
from torch.nn import Linear

from torch_geometric.nn import GATConv, GCNConv, global_add_pool


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3, 16)
        self.conv2 = GCNConv(16, 7)

    def forward(self, g, **kwargs):
        x = self.conv1(g.x, g.edge_index)
        x = F.relu(x)
        x = self.conv2(x, g.edge_index)
        return x.log_softmax(dim=1)


class GAT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GATConv(3, 16, heads=2, concat=True)
        self.conv2 = GATConv(2 * 16, 7, heads=2, concat=False)

    def forward(self, g, **kwargs):
        x = self.conv1(g.x, g.edge_index)
        x = F.relu(x)
        x = self.conv2(x, g.edge_index)
        return x.log_softmax(dim=1)


class GNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3, 16)
        self.conv2 = GCNConv(16, 16)
        self.lin = Linear(16, 7)

    def forward(self, g, batch=None, **kwargs):
        # edge_attr is for checking if explain can
        # pass arguments in the right oder
        x = self.conv1(g.x, g.edge_index)
        x = F.relu(x)
        x = self.conv2(x, g.edge_index)
        x = global_add_pool(x, batch)
        x = self.lin(x)
        return x.log_softmax(dim=1)


return_types = ['log_prob', 'regression']
feat_mask_types = ['individual_feature', 'scalar', 'feature']
