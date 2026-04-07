import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader, NeighborLoader
from torch_geometric.nn import KerGNNConv, Linear, global_add_pool
from torch_geometric.seed import seed_everything

seed_everything(42)
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'TU')
dataset = TUDataset(path, name='PROTEINS', use_node_attr=True)
dataset = dataset.shuffle()

train_size = 900
train_dataset = dataset[:train_size]
test_dataset = dataset[train_size:]

train_loader = DataLoader(train_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)


class Net(torch.nn.Module):
    def __init__(self, features_dim, n_classes, hidden_channels, kernel, power,
                 dropout_rate, size_graph_filter, no_norm):
        assert len(hidden_channels) == len(size_graph_filter) + 1
        super().__init__()
        self.no_norm = no_norm
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(p=dropout_rate)
        self.num_layers = len(hidden_channels) - 1

        self.linear_transform = Linear(features_dim, 16)
        self.ker_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for layer in range(self.num_layers):
            self.ker_layers.append(
                KerGNNConv(hidden_channels[layer], hidden_channels[layer + 1],
                           kernel=kernel, power=power,
                           size_graph_filter=size_graph_filter[layer],
                           dropout=dropout_rate))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels[layer + 1]))

        self.linear = nn.ModuleList()
        for layer in range(self.num_layers + 1):
            self.linear.append(
                Linear(features_dim if layer == 0 else hidden_channels[layer],
                       n_classes))

    def forward(self, x, edge_index, batch):
        _, counts = torch.unique(batch, return_counts=True)
        hidden_rep = [x]
        h = F.relu(self.linear_transform(x))

        for layer in range(self.num_layers):
            h = self.ker_layers[layer](h, edge_index)
            h = F.relu(self.batch_norms[layer](h))
            hidden_rep.append(h)

        score_over_layer = 0
        for layer, h in enumerate(hidden_rep):
            pooled_h = global_add_pool(h, batch)
            if not self.no_norm:
                norm = counts.unsqueeze(1).repeat(1, pooled_h.shape[1])
                pooled_h = pooled_h / norm
            score_over_layer += F.dropout(self.linear[layer](pooled_h),
                                          self.dropout_rate,
                                          training=self.training)

        return score_over_layer


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(features_dim=dataset.num_features, n_classes=dataset.num_classes,
            hidden_channels=[16, 32], kernel='rw', power=1, dropout_rate=0.4,
            size_graph_filter=[6], no_norm=False)

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)


def train():
    model.train()
    total_loss = 0.
    for data in train_loader:
        loader = NeighborLoader(data, num_neighbors=[10],
                                batch_size=data.num_nodes, shuffle=False)
        for subgraph_data in loader:
            data = data.to(device)
            subgraph_data = subgraph_data.to(device)
            optimizer.zero_grad()
            out = model(subgraph_data.x, subgraph_data.edge_index, data.batch)
            loss = F.cross_entropy(out, data.y)
            loss.backward()
            total_loss += loss.item() * data.num_graphs
            optimizer.step()
    return total_loss / len(train_dataset)


@torch.no_grad()
def test(loader):
    model.eval()
    total_correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        pred = out.max(dim=1)[1]
        total_correct += pred.eq(data.y).sum().item()
    return total_correct / len(loader.dataset)


best_train_acc = 0
best_test_acc = 0
for epoch in range(200):
    loss = train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)

    if test_acc > best_test_acc:
        best_test_acc = test_acc
        best_train_acc = train_acc

    if epoch % 20 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
              f'Train: {train_acc:.4f}, Test: {test_acc:.4f}')

print(f"Best train acc: {best_train_acc}")
print(f"Best test acc: {best_test_acc}")
