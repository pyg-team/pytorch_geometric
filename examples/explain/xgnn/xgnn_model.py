from torch_geometric.nn import GCNConv
import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

def reset_parameters(module):
    if isinstance(module, torch.nn.Linear):
        module.reset_parameters()
        print(module.parameters())

### GCN to predict graph property
class GCN_Graph(torch.nn.Module):
    def __init__(self, input_dim, output_dim, dropout, emb = False):
        super(GCN_Graph, self).__init__()

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
      self.fc1.reset_parameters()
      self.fc2.reset_parameters()

    def forward(self, batched_data):
        # Extract important attributes of our mini-batch
        x, edge_index, batch = batched_data.x, batched_data.edge_index, batched_data.batch

        for i in range(len(self.convs)):
            x = F.relu(self.convs[i](x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        # print(x)
        # x = torch.mean(x, 1)
        # print(x)
        x = self.pool(x, batch)
        # print(x)

        x = F.relu(self.fc1(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc2(x)
        # print("size of x")
        # print(x.size())
        x = F.softmax(x, dim=1)
        # print(x.size())
        return x