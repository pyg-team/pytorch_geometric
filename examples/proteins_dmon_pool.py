from math import ceil

import torch
import torch.nn.functional as F
from torch.nn import Linear

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import DenseGraphConv, DMonPooling, GCNConv
from torch_geometric.utils import to_dense_adj, to_dense_batch

dataset = TUDataset(root='/tmp/PROTEINS', name='PROTEINS').shuffle()
avg_num_nodes = int(dataset.data.x.size(0) / len(dataset))
n = (len(dataset) + 9) // 10
test_dataset = dataset[:n]
val_dataset = dataset[n:2 * n]
train_dataset = dataset[2 * n:]
test_loader = DataLoader(test_dataset, batch_size=20)
val_loader = DataLoader(val_dataset, batch_size=20)
train_loader = DataLoader(train_dataset, batch_size=20)


class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=32):
        super().__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels)
        num_nodes = ceil(0.5 * avg_num_nodes)
        self.pool1 = DMonPooling([hidden_channels, hidden_channels], num_nodes)

        self.conv2 = DenseGraphConv(hidden_channels, hidden_channels)
        num_nodes = ceil(0.5 * num_nodes)
        self.pool2 = DMonPooling([hidden_channels, hidden_channels], num_nodes)

        self.conv3 = DenseGraphConv(hidden_channels, hidden_channels)

        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch=None):
        x = self.conv1(x, edge_index).relu()
        x, mask = to_dense_batch(x, batch)
        adj = to_dense_adj(edge_index, batch)

        _, x, adj, sp1, o1, c1 = self.pool1(x, adj, mask)

        x = self.conv2(x, adj).relu()
        _, x, adj, sp2, o2, c2 = self.pool2(x, adj)

        x = self.conv3(x, adj)

        x = x.mean(dim=1)
        x = self.lin1(x).relu()
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1), sp1 + sp2 + o1 + o2 + c1 + c2


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(dataset.num_features, dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)


def train(train_loader):
    model.train()
    loss_all = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out, tot_loss = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(out, data.y.view(-1)) + tot_loss
        loss.backward()
        loss_all += data.y.size(0) * loss.item()
        optimizer.step()
    return loss_all / len(train_dataset)


@torch.no_grad()
def test(loader):
    model.eval()
    correct = 0
    loss_all = 0

    for data in loader:
        data = data.to(device)
        pred, tot_loss = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(pred, data.y.view(-1)) + tot_loss
        loss_all += data.y.size(0) * loss.item()
        correct += pred.max(dim=1)[1].eq(data.y.view(-1)).sum().item()

    return loss_all / len(loader.dataset), correct / len(loader.dataset)


for epoch in range(100):
    train_loss = train(train_loader)
    _, train_acc = test(train_loader)
    val_loss, val_acc = test(val_loader)
    test_loss, test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.3f}, '
          f'Train Acc: {train_acc:.3f}, Val Loss: {val_loss:.3f}, '
          f'Val Acc: {val_acc:.3f}, Test Loss: {test_loss:.3f}, '
          f'Test Acc: {test_acc:.3f}')
