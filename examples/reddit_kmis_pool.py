import os.path as osp

import torch
import torch.nn.functional as F

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, KMISPooling
from torch_geometric.nn import global_add_pool as gap
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.transforms import Constant

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data',
                'REDDIT-BINARY')
dataset = TUDataset(path, name='REDDIT-BINARY', transform=Constant())
dataset = dataset.shuffle()
n = len(dataset) // 10
test_dataset = dataset[:2 * n]
valid_dataset = dataset[2 * n:3 * n]
train_dataset = dataset[3 * n:]
test_loader = DataLoader(test_dataset, batch_size=60)
valid_loader = DataLoader(valid_dataset, batch_size=60)
train_loader = DataLoader(train_dataset, batch_size=60)


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()

        channels = 128
        self.conv1 = GCNConv(dataset.num_features, channels)
        self.pool1 = KMISPooling(channels, k=1)
        self.conv2 = GCNConv(channels, 2 * channels)
        self.pool2 = KMISPooling(2 * channels, k=1)
        self.conv3 = GCNConv(2 * channels, 4 * channels)

        self.lin1 = torch.nn.Linear(8 * channels, 2 * channels)
        self.lin2 = torch.nn.Linear(2 * channels, dataset.num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, edge_attr, batch, _, _ = self.pool1(
            x, edge_index, None, batch)

        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x, edge_index, edge_attr, batch, _, _ = self.pool2(
            x, edge_index, edge_attr, batch)

        x = F.relu(self.conv3(x, edge_index, edge_attr))
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.log_softmax(self.lin2(x), dim=-1)

        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)


def train():
    model.train()

    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(train_dataset)


def test(loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        pred = model(data).max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


best_valid_acc = test_acc = 0.
for epoch in range(1, 71):
    loss = train()
    train_acc = test(train_loader)
    valid_acc = test(valid_loader)

    if valid_acc > best_valid_acc:
        best_valid_acc = valid_acc
        test_acc = test(test_loader)

    print(f'Epoch: {epoch:03d}, Loss: {loss:.5f}, Train Acc: {train_acc:.5f}, '
          f'Valid Acc: {valid_acc:.5f}')

print(f'\nTest Acc: {test_acc:.5f}\n')
