from torch_geometric.nn import GCNConv
import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch.nn.parameter import Parameter
import math 

# def reset_parameters(module):
#     if isinstance(module, torch.nn.Linear):
#         module.reset_parameters()
#         print(module.parameters())

### GCN to predict graph property
class GCN_Graph(torch.nn.Module):
    def __init__(self, input_dim, output_dim, dropout, emb = False):
        super(GCN_Graph, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.dropout = dropout
        self.convs = torch.nn.ModuleList([GCNConv(in_channels = input_dim, out_channels = 32),
                                          GCNConv(in_channels = 32,        out_channels = 48),
                                          GCNConv(in_channels = 48,        out_channels = 64)])

        self.pool = global_mean_pool # global averaging to obtain graph representation

        # self.post_mp = torch.nn.Sequential(torch.nn.Linear(64, 32),
        #                                    torch.nn.Dropout(self.dropout),
        #                                    torch.nn.Linear(32, output_dim))

        self.fc1 = torch.nn.Linear(64, 32)
        self.fc2 = torch.nn.Linear(32, output_dim)

        self.loss = torch.nn.BCEWithLogitsLoss()
        self.reset_parameters()

    def reset_parameters(self):
      for conv in self.convs:
          conv.reset_parameters()
          stdv = 1. / math.sqrt(conv.lin.weight.size(1))
          torch.nn.init.uniform_(conv.lin.weight, -stdv, stdv)
          
          conv.bias = Parameter(torch.FloatTensor(conv.out_channels))
          conv.bias.data.uniform_(-stdv, stdv)
          
      self.fc1.reset_parameters()
      self.fc2.reset_parameters()

    def forward(self, batched_data):
        # Extract important attributes of our mini-batch
        x, edge_index, batch = batched_data.x, batched_data.edge_index, batched_data.batch

        for i in range(len(self.convs)):
            x = F.relu(self.convs[i](x, edge_index))
            if i < len(self.convs) - 1: # do not apply dropout on last layer
                x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.pool(x, batch)
        
        x = F.relu(self.fc1(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc2(x)
        # x = F.softmax(x, dim=1)
        x = F.sigmoid(x)
        return x