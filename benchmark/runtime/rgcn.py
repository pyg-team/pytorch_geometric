import torch
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv


class RGCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_relations):
        super(RGCN, self).__init__()
        self.conv1 = RGCNConv(in_channels, 16, num_relations, num_bases=30)
        self.conv2 = RGCNConv(16, out_channels, num_relations, num_bases=30)

    def forward(self, data):
        edge_index, edge_type = data.edge_index, data.edge_type
        x = F.relu(self.conv1(None, edge_index, edge_type))
        x = self.conv2(x, edge_index, edge_type)
        return F.log_softmax(x, dim=1)
