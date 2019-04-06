import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GraphConv, global_mean_pool, graclus, max_pool
from torch_geometric.data import Batch


class Graclus(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden):
        super(Graclus, self).__init__()
        self.conv1 = GraphConv(dataset.num_features, hidden, aggr='mean')
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GraphConv(hidden, hidden, aggr='mean'))
        self.lin1 = Linear(num_layers * hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        xs = [global_mean_pool(x, batch)]
        for i, conv in enumerate(self.convs):
            x = F.relu(conv(x, edge_index))
            xs += [global_mean_pool(x, batch)]
            if i % 2 == 0:
                cluster = graclus(edge_index, num_nodes=x.size(0))
                data = Batch(x=x, edge_index=edge_index, batch=batch)
                data = max_pool(cluster, data)
                x, edge_index, batch = data.x, data.edge_index, data.batch
        x = torch.cat(xs, dim=1)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__
